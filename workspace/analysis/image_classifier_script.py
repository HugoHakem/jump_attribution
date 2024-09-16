#!/usr/bin/env python
# coding: utf-8
import os
import polars as pl
import pandas as pd

import zarr
import numcodecs
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

from itertools import groupby, starmap, product
from more_itertools import unzip

from data_split import StratifiedGroupKFold_custom
from sklearn.model_selection import StratifiedShuffleSplit

from jump_portrait.fetch import get_jump_image
from jump_portrait.utils import batch_processing, parallel

from collections.abc import Callable, Iterable
from typing import List

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import lightning as L


from parallel_training import run_train
import conv_model
import custom_dataset
from lightning_parallel_training import LightningModel
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint

from pathlib import Path

# # 1) Loading images and create a pytorch dataset

# ## a) Load Images using Jump_portrait

# In[2]:

file_directory = Path("/home/hhakem/projects/counterfactuals_projects/workspace/analysis/figures")
metadata_pre = pl.read_csv("target2_eq_moa2_metadata")


def try_function(f: Callable):
    '''
    Wrap a function into an instance which will Try to call the function:
        If it success, return a tuple of function parameters + its results
        If it fails, return the function parameters
    '''
    # This assume parameters are packed in a tuple
    def batched_fn(*item, **kwargs):
        try:
            result = (*item, f(*item, **kwargs))
            
        except:
            result = item

        return result    
    return batched_fn


# ### ii) function to overcome turn get_jump_image_iter compatible with list and load data in a threaded fashion

def get_jump_image_iter(metadata: pl.DataFrame, channel: List[str],
                        site:List[str], correction:str=None) -> (pl.DataFrame, List[tuple]): 
    '''
       Load jump image associated to metadata in a threaded fashion.
        ----------
    Parameters: 
        metadata(pl.DataFrame): must have the shape (Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well")
        channel(List[str]): list of channel desired
            Must be in ['DNA', 'ER', 'AGP', 'Mito', 'RNA']
        site(List[str]): list of site desired
            For compound, must be in ['1' - '6']
            For ORF, CRISPR, must be in ['1' - '9']
        correction(str): Must be 'Illum' or None
        ----------
    Return:
        features(pl.DataFrame): DataFrame collecting the metadata, channel, site, correction + the image
        work_fail(List(tuple): List collecting tuple of metadata which failed to load an image
        
    '''
    iterable = [(*metadata.row(i), ch, s, correction)
               for i in range(metadata.shape[0]) for s in site for ch in channel]
    img_list = parallel(iterable, batch_processing(try_function(get_jump_image)))
    
    img_list = sorted(img_list, key=lambda x: len(x))
    fail_success = {k: list(g) for k, g in groupby(img_list, key=lambda x: len(x))}
    if len(fail_success) == 1:
        img_success = list(fail_success.values())[0]
        work_fail = []
    else: 
        work_fail, img_success = fail_success.values()
    features = pl.DataFrame(img_success, 
                               schema=["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well",
                                        "channel", "site", "correction",
                                        "img"])
    return features, work_fail

if not os.path.exists(Path("image_dataset/imgs_labels_groups.zarr")):
    channel = ['AGP', 'DNA', 'ER', 'Mito', 'RNA']
    features_pre, work_fail = get_jump_image_iter(metadata_pre.select(pl.col(["Metadata_Source", "Metadata_Batch",
                                                                              "Metadata_Plate", "Metadata_Well"])),
                                                  channel=channel,#, 'ER', 'AGP', 'Mito', 'RNA'],
                                                  site=[str(i) for i in range(1, 7)],
                                                  correction=None) #None, 'Illum'


    # ### iii) Add 'site' 'channel' and filter out sample which could not be load (using join)

    correct_index = (features_pre.select(pl.all().exclude("correction", "img"))
                     .sort(by=["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well", "site", "channel"])
                     .group_by(by=["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well", "site"], maintain_order=True)
                     .all()
                     .with_columns(pl.arange(0,len(features_pre)//len(channel)).alias("ID")))

    metadata = metadata_pre.select(pl.all().exclude("ID")).join(correct_index,
                                                                on=["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well"],
                                                                how="inner").select("ID", pl.all().exclude("ID")).sort("ID")

    features = features_pre.join(metadata.select(pl.col(["Metadata_Source", "Metadata_Batch",
                                                         "Metadata_Plate", "Metadata_Well", "site",  "ID"])),
                                 on=["Metadata_Source", "Metadata_Batch", "Metadata_Plate",
                                     "Metadata_Well", "site"],
                                 how="inner").sort(by=["ID", "channel"])

    # ## #a) crop image and stack them
    def crop_square_array(x, des_size):
        h, w = x.shape
        h_off, w_off = (h-des_size)//2, (w-des_size)//2
        return x[h_off:des_size+h_off,w_off:des_size+w_off]


    shape_image = list(map(lambda x: x.shape, features["img"].to_list()))
    shape_image.sort(key=lambda x:x[0])
    img_crop = list(map(lambda x: crop_square_array(x, shape_image[0][0]), features["img"].to_list()))
    img_stack = np.array([np.stack([item[1] for item in tostack])
                 for idx, tostack in groupby(zip(features["ID"].to_list(), img_crop), key=lambda x: x[0])])

    # ## b) Encode moa into moa_id
    metadata_df = metadata.to_pandas().set_index(keys="ID")
    metadata_df = metadata_df.assign(moa_id=LabelEncoder().fit_transform(metadata_df["moa"]))
    labels, groups = metadata_df["moa_id"].values, metadata_df["Metadata_InChIKey"].values

    # ## c) Clip image
    clip = (1, 99)
    min_clip_img, max_clip_img = np.percentile(img_stack, clip, axis=(2,3), keepdims=True)
    clip_image = np.clip(img_stack,
                         min_clip_img,
                         max_clip_img)

    # ## d) Split image in 4
    def slice_image(img, labels, groups):
        image_size = img.shape[-1]
        small_image = np.vstack([img[:, :, :image_size//2, :image_size//2],
                                 img[:, :, :image_size//2, image_size//2:],
                                 img[:, :, image_size//2:, :image_size//2],
                                 img[:, :, image_size//2:, image_size//2:]])
        small_labels = np.hstack([labels for i in range(4)])
        small_groups = np.hstack([groups for i in range(4)])
        return small_image, small_labels, small_groups
    small_image, small_labels, small_groups = slice_image(clip_image, labels, groups)

    store = zarr.DirectoryStore(Path("image_dataset/imgs_labels_groups.zarr"))
    root = zarr.group(store=store, )
    # Save datasets into the group
    root.create_dataset('imgs', data=small_image, overwrite=True, chunks=(1, 1, *small_image.shape[2:]))
    root.create_dataset('labels', data=small_labels, overwrite=True, chunks=1)
    root.create_dataset('groups', data=small_groups, dtype=object, object_codec=numcodecs.JSON(), overwrite=True, chunks=1)


# ## e) Create the pytorch dataset with respect to kfold split with train val and test set
image_dataset = zarr.open(Path("image_dataset/imgs_labels_groups.zarr"))
kfold_train_test = list(StratifiedGroupKFold_custom(random_state=42).split(None, image_dataset["labels"][:], image_dataset["groups"][:]))
kfold_train_val_test = list(starmap(lambda train, val_test: (train,
                                                             *list(starmap(lambda val, test: (val_test[val], val_test[test]),
                                                                           StratifiedShuffleSplit(n_splits=1, random_state=42, test_size=0.5).split(
                                                                               np.zeros(len(val_test)), image_dataset["labels"].oindex[val_test])))[0]),
                                    kfold_train_test))
# ## #i) Transformation applied to train split
transform_train = v2.RandomApply([v2.RandomVerticalFlip(p=0.5),
                                  v2.RandomChoice([v2.Lambda(lambda img: v2.functional.rotate(img, angle=0)),
                                                   v2.Lambda(lambda img: v2.functional.rotate(img, angle=90)),
                                                   v2.Lambda(lambda img: v2.functional.rotate(img, angle=180)),
                                                   v2.Lambda(lambda img: v2.functional.rotate(img, angle=270))])],
                                 p=1)

fold_L = np.arange(5)
channel = ["AGP"] # "DNA"]
channel.sort()
map_channel = {ch: i for i, ch in enumerate(["AGP", "DNA", "ER", "Mito", "RNA"])}
id_channel = np.array([map_channel[ch] for ch in channel])
imgs_path = Path("image_dataset/imgs_labels_groups.zarr")
dataset_fold = {i: {"train": custom_dataset.ImageDataset(imgs_path,
                                                         channel=id_channel,
                                                         fold_idx=kfold_train_val_test[i][0],
                                                         img_transform=v2.Compose([v2.Lambda(lambda img: torch.tensor(img, dtype=torch.float32)),
                                                                                   transform_train]),
                                                         label_transform=lambda label: torch.tensor(label, dtype=torch.long)),
                    "val": custom_dataset.ImageDataset(imgs_path,
                                                       channel=id_channel,
                                                       fold_idx=kfold_train_val_test[i][1],
                                                       img_transform=lambda img: torch.tensor(img, dtype=torch.float32),
                                                       label_transform=lambda label: torch.tensor(label, dtype=torch.long)),
                    "test": custom_dataset.ImageDataset(imgs_path,
                                                       channel=id_channel,
                                                       fold_idx=kfold_train_val_test[i][2],
                                                       img_transform=lambda img: torch.tensor(img, dtype=torch.float32),
                                                       label_transform=lambda label: torch.tensor(label, dtype=torch.long))}
                for i in fold_L}

# ### i) Memory usage per fold

# def fold_memory_usage(fold:int, split:str ="train", batch_size:int=None):
#     total_size = 0
#     for i in range(len(dataset_fold[fold][split])):
#         sample = dataset_fold[fold][split][i]
#         sample_size = 0
#         if isinstance(sample, tuple):
#             for item in sample:
#                 if isinstance(item, torch.Tensor):
#                     sample_size += item.element_size() * item.nelement()
#         total_size += sample_size
#     if batch_size is not None:
#         normalizer = batch_size / len(dataset_fold[fold][split])
#     else:
#         normalizer = 1
#     print(f"Total fold {fold} size for {split} set with {batch_size} "+
#           f"batch_size: {normalizer * total_size / (1024 ** 2):.2f} MB")

# fold_memory_usage(0, "train", None)


# # # 2) Model

# # ## a) Receptive field calculation
# # Receptive field are important to visualise what information of the original image is convolve to get end features
# # Computation of the receptive field is base don this [article](https://distill.pub/2019/computing-receptive-fields/).

# def compute_receptive_field(model):
#     dict_module = dict(torch.fx.symbolic_trace(model).named_modules())
#     def extract_kernel_stride(module):
#         try:
#             return (module.kernel_size[0] if type(module.kernel_size) == tuple else module.kernel_size,
#                     module.stride[0] if type(module.stride) == tuple else module.stride)
#         except:
#             return None

#     k, s = list(map(lambda x: np.array(list(x)),
#                     unzip([x for x in list(map(extract_kernel_stride, list(dict_module.values()))) if x is not None])))
#     return ((k-1) * np.concatenate((np.array([1]), s.cumprod()[:-1]))).sum() + 1

# def compute_receptive_field_recursive(model):
#     dict_module = dict(torch.fx.symbolic_trace(model).named_modules())
#     def extract_kernel_stride(module):
#         try:
#             return (module.kernel_size[0] if type(module.kernel_size) == tuple else module.kernel_size,
#                     module.stride[0] if type(module.stride) == tuple else module.stride)
#         except:
#             return None

#     k, s = list(map(lambda x: np.array(list(x))[::-1],
#                     unzip([x for x in list(map(extract_kernel_stride, list(dict_module.values()))) if x is not None])))

#     res = [1, k[0]]
#     for i in range(1, len(k)):
#         res.append(s[i]*res[i]+k[i]-s[i])
#     return res

# # ## b) Memory usage calculation
# # Memory is the bottleneck in GPU training so knowing the size of the model is important

# def model_memory_usage(model):
#     param_size = 0
#     for param in model.parameters():
#         param_size += param.nelement() * param.element_size()
#     buffer_size = 0
#     for buffer in model.buffers():
#         buffer_size += buffer.nelement() * buffer.element_size()

#     size_all_mb = (param_size + buffer_size) / 1024**2
#     print('model size: {:.2f}MB'.format(size_all_mb))
#     free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
#     print(f"Free memory in cuda 0 before model load: {free_mem / 1024**2:.2f} MB")
#     free_mem = torch.cuda.get_device_properties(1).total_memory - torch.cuda.memory_allocated(1)
#     print(f"Free memory in cuda 1 before model load: {free_mem / 1024**2:.2f} MB")


# vgg = conv_model.VGG(img_depth=1,
#           img_size=485,
#           lab_dim=7,
#           n_conv_block=4,
#           n_conv_list=[1 for _ in range(4)],
#           n_lin_block=3)

# vgg_ch = conv_model.VGG_ch(
#           img_depth=1,
#           img_size=485,
#           lab_dim=7,
#           conv_n_ch=64,
#           n_conv_block=6,
#           n_conv_list=[2, 2, 2, 2, 4, 4],
#           n_lin_block=3,
#           p_dropout=0.2)

# print(f'recursive receptive field from very end to start: {compute_receptive_field_recursive(vgg_ch)}')
# recept_field = compute_receptive_field(vgg_ch)
# print(f'recursive receptive field at the start: {recept_field}')
# #model_memory_usage(vgg)



# #Visualisation of the chosen kernel size relative to the image size
# fig, axes = plt.subplots(1, 1, figsize=(30,30))
# axes.imshow(dataset_fold[0]["train"][1][0][0])
# rect = patches.Rectangle((130, 280), recept_field, recept_field, linewidth=7, edgecolor='r', facecolor='none')
# axes.add_patch(rect)
# fig.savefig(file_directory / "kernel_vgg_vs_small_image_size")
# plt.close()




os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# # Lightning Training
fold = 0
tb_logger = pl_loggers.TensorBoardLogger(save_dir=Path("logs"), name="VGG_image_AGP_DNA_ER")
checkpoint_callback = ModelCheckpoint(dirpath=Path("lightning_checkpoint_log"),
                                      filename=f"VGG_image_AGP_DNA_ER_fold_{fold}"+"{epoch}-{train_acc:.2f}-{val_acc:.2f}",
                                      every_n_epochs=10)

torch.set_float32_matmul_precision('medium') #try 'high')
seed_everything(42, workers=True)
# lit_model = LightningModel(conv_model.VGG,
#                            model_param=(1, #img_depth
#                                         485, #img_size
#                                         7, #lab_dim
#                                         5, #n_conv_block
#                                         [1, 1, 1, 1, 1], #n_conv_list
#                                         3),
#                            lr=5e-4,
#                            weight_decay=5e-3,
#                            max_epoch=1,
#                            n_class=7)

max_epoch = 1
lit_model = LightningModel(conv_model.VGG_ch,
                           model_param=(len(channel), #img_depth
                                        485, #img_size
                                        7, #lab_dim
                                        32, #conv_n_ch
                                        6, #n_conv_block
                                        [1, 1, 2, 2, 3, 3], #n_conv_list
                                        3, #n_lin_block
                                        0.2), #p_dropout
                           lr=5e-4,
                           weight_decay=0,
                           max_epoch=max_epoch,
                           n_class=7)

trainer = L.Trainer(#default_root_dir="./lightning_checkpoint_log/",
                    accelerator="gpu",
                    devices=2,
                    strategy="ddp",
                    max_epochs=max_epoch,
                    logger=tb_logger,
                    #profiler="simple",
                    num_sanity_val_steps=0, #to use only if you know the trainer is working !
                    callbacks=[checkpoint_callback],
                    #enable_checkpointing=False,
                    enable_progress_bar=True
                    )

import resource
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

trainer.fit(lit_model, DataLoader(dataset_fold[0]["train"], batch_size=128, num_workers=1, shuffle=True, persistent_workers=True),
            DataLoader(dataset_fold[0]["val"], batch_size=128, num_workers=1, persistent_workers=True))