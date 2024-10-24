#!/usr/bin/env python
# coding: utf-8
import os
import polars as pl
import pandas as pd
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

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
from torch.utils.data import Dataset, DataLoader, BatchSampler
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import lightning as L


from parallel_training import run_train
import conv_model
import custom_dataset
from lightning_parallel_training import LightningModelV2, LightningGANV2, LightningStarGANV2
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.combined_loader import CombinedLoader

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from pathlib import Path
import xarray as xr

# # 1) Loading images and create a pytorch dataset

# ## a) Load Images using Jump_portrait

# In[2]:

fig_directory = Path("/home/hhakem/projects/counterfactuals_projects/workspace/analysis/figures")
metadata_pre = pl.read_csv("target2_eq_moa2_active_metadata")


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

if not os.path.exists(Path("image_active_dataset/imgs_labels_groups.zarr")):
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


    # shape_image = list(map(lambda x: x.shape, features["img"].to_list()))
    # shape_image.sort(key=lambda x:x[0])
    img_crop = list(map(lambda x: crop_square_array(x, des_size=896), features["img"].to_list())) #shape_image[0][0]
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
    clip_image = (clip_image - min_clip_img) / (max_clip_img - min_clip_img)

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

    store = zarr.DirectoryStore(Path("image_active_dataset/imgs_labels_groups.zarr"))
    root = zarr.group(store=store)
    # Save datasets into the group
    root.create_dataset('imgs', data=small_image, overwrite=True, chunks=(1, 1, *small_image.shape[2:]))
    root.create_dataset('labels', data=small_labels, overwrite=True, chunks=1)
    root.create_dataset('groups', data=small_groups, dtype=object, object_codec=numcodecs.JSON(), overwrite=True, chunks=1)


# ## e) Create the pytorch dataset with respect to kfold split with train val and test set
image_dataset = zarr.open(Path("image_active_dataset/imgs_labels_groups.zarr"))
kfold_train_test = list(StratifiedGroupKFold_custom(random_state=42).split(None, image_dataset["labels"][:], image_dataset["groups"][:]))
kfold_train_val_test = list(starmap(lambda train, val_test: (train,
                                                             *list(starmap(lambda val, test: (val_test[val], val_test[test]),
                                                                           StratifiedShuffleSplit(n_splits=1, random_state=42, test_size=0.5).split(
                                                                               np.zeros(len(val_test)), image_dataset["labels"].oindex[val_test])))[0]),
                                    kfold_train_test))
# ## #i) Transformation applied to train split
img_transform_train = v2.RandomApply([v2.RandomVerticalFlip(p=0.5),
                                      v2.RandomChoice([v2.Lambda(lambda img: v2.functional.rotate(img, angle=0)),
                                                       v2.Lambda(lambda img: v2.functional.rotate(img, angle=90)),
                                                       v2.Lambda(lambda img: v2.functional.rotate(img, angle=180)),
                                                       v2.Lambda(lambda img: v2.functional.rotate(img, angle=270))])],
                                     p=1)

fold_L = np.arange(5)
channel = ["AGP","DNA", "ER"]#, "Mito"]#, "RNA"]
channel.sort()
map_channel = {ch: i for i, ch in enumerate(["AGP", "DNA", "ER", "Mito", "RNA"])}
id_channel = np.array([map_channel[ch] for ch in channel])
imgs_path = Path("image_active_dataset/imgs_labels_groups.zarr")

def create_dataset_fold(Dataset, imgs_path, id_channel, kfold_train_val_test, img_transform_train):
    return {i: {"train": Dataset(imgs_path,
                                 channel=id_channel,
                                 fold_idx=kfold_train_val_test[i][0],
                                 img_transform=v2.Compose([v2.Lambda(lambda img:
                                                                     torch.tensor(img, dtype=torch.float32)),
                                                           img_transform_train,
                                                           v2.Normalize(mean=len(channel)*[0.5],
                                                                                std=len(channel)*[0.5])]),
                                 label_transform=lambda label: torch.tensor(label, dtype=torch.long)),
                "val": Dataset(imgs_path,
                               channel=id_channel,
                               fold_idx=kfold_train_val_test[i][1],
                               img_transform=v2.Compose([v2.Lambda(lambda img:
                                                                     torch.tensor(img, dtype=torch.float32)),
                                                        v2.Normalize(mean=len(channel)*[0.5],
                                                                     std=len(channel)*[0.5])]),
                               label_transform=lambda label: torch.tensor(label, dtype=torch.long)),
                "test": Dataset(imgs_path,
                                channel=id_channel,
                                fold_idx=kfold_train_val_test[i][2],
                                img_transform=v2.Compose([v2.Lambda(lambda img:
                                                                     torch.tensor(img, dtype=torch.float32)),
                                                          v2.Normalize(mean=len(channel)*[0.5],
                                                                       std=len(channel)*[0.5])]),
                                label_transform=lambda label: torch.tensor(label, dtype=torch.long))}
            for i in fold_L}


dataset_fold = create_dataset_fold(custom_dataset.ImageDataset, imgs_path, id_channel, kfold_train_val_test,
                                   img_transform_train)
dataset_fold_ref = create_dataset_fold(custom_dataset.ImageDataset_Ref, imgs_path, id_channel, kfold_train_val_test,
                                       img_transform_train)
# ### I) Memory usage per fold

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
# fig.savefig(fig_directory / "kernel_vgg_vs_small_image_size")
# plt.close()


'''
classifier Training
'''

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["NCCL_P2P_DISABLE"] = "1"
# # # Lightning Training
# fold = 0
# tb_logger = pl_loggers.TensorBoardLogger(save_dir=Path("logs"), name="VGG_image_active")
# checkpoint_callback = ModelCheckpoint(dirpath=Path("lightning_checkpoint_log"),
#                                       filename=f"VGG_image_active_fold_{fold}"+"{epoch}-{train_acc:.2f}-{val_acc:.2f}",
#                                       save_top_k=1,
#                                       monitor="val_acc",
#                                       mode="max",
#                                       every_n_epochs=1)

# torch.set_float32_matmul_precision('medium') #try 'high')
# seed_everything(42, workers=True)


# max_epoch = 80
# lit_model = LightningModelV2(conv_model.VGG_ch,
#                              model_param=(len(channel), #img_depth
#                                          448, #img_size
#                                          4, #lab_dim
#                                          16, #conv_n_ch 32
#                                          7, #n_conv_block 6
#                                          [1, 1, 2, 2, 3, 3, 3], #n_conv_list
#                                          3, #n_lin_block
#                                          0.2), #p_dropout
#                              lr=5e-4,
#                              weight_decay=0,
#                              max_epoch=max_epoch,
#                              n_class=4)

# trainer = L.Trainer(#default_root_dir="./lightning_checkpoint_log/",
#                     accelerator="gpu",
#                     devices=2,
#                     strategy="ddp_notebook",
#                     max_epochs=max_epoch,
#                     logger=tb_logger,
#                     #profiler="simple",
#                     num_sanity_val_steps=0, #to use only if you know the trainer is working !
#                     callbacks=[checkpoint_callback],
#                     #enable_checkpointing=False,
#                     enable_progress_bar=False
#                     #deterministic=True #using along with torchmetric: it slow down process as it apparently rely
#                     #on some cumsum which is not permitted in deterministic on GPU and then transfer to CPU
#                     )

# import resource
# soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

# trainer.fit(lit_model, DataLoader(dataset_fold[0]["train"], batch_size=128, num_workers=1, shuffle=True, persistent_workers=True),
#             DataLoader(dataset_fold[0]["val"], batch_size=128, num_workers=1, persistent_workers=True))


'''
GANs Training
'''


# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["NCCL_P2P_DISABLE"] = "1"

# # Some parameter definition
# fold=0
# img_size = 448
# num_domains = 4 #n_class
# max_epoch = 30
# latent_dim = 16
# style_dim = 64

# # # Lightning Training
# tb_logger = pl_loggers.TensorBoardLogger(save_dir=Path("logs"), name="StarGANv2_image_active")
# checkpoint_callback = ModelCheckpoint(dirpath=Path("lightning_checkpoint_log"),
#                                       filename=f"StarGANv2_image_active_fold_{fold}_"+"{epoch}-{step}", #-{train_acc_true:.2f}-{train_acc_fake:.2f}",
#                                       #save_top_k=1,
#                                       #monitor="val_acc",
#                                       #mode="max",
#                                       every_n_train_steps=50)
#                                       #every_n_epochs=1)

# torch.set_float32_matmul_precision('medium') #try 'high')
# seed_everything(42, workers=True)

# lit_model = LightningStarGANV2(
#     conv_model.Generator, # generator
#     conv_model.MappingNetwork, # mapping_network
#     conv_model.StyleEncoder, # style_encoder
#     conv_model.Discriminator, # discriminator
#     {"num_channels": len(channel), "dim_in": 64, "style_dim": style_dim, "num_block": 4, "max_conv_dim": 512}, # generator_param
#     {"latent_dim": latent_dim, "style_dim": 64, "num_domains": num_domains}, # mapping_network
#     {"img_size": img_size, "num_channels": len(channel), "num_domains": num_domains, "dim_in": 64, "style_dim": style_dim,
#      "num_block": 4, "max_conv_dim": 512}, # style_encoder_param
#     {"img_size": img_size, "num_channels": len(channel), "num_domains": num_domains, "dim_in": 64, "style_dim": style_dim,
#      "num_block": 4, "max_conv_dim": 512}, # discriminator_param,
#     {"lr": 1e-4, "betas": (0, 0.99)}, # adam_param_g G
#     {"lr": 1e-6, "betas": (0, 0.99)}, # adam_param_m F
#     {"lr": 1e-4, "betas": (0, 0.99)}, # adam_param_s E
#     {"lr": 1e-4, "betas": (0, 0.99)}, # adam_param_d D
#     {"lambda_cyc": 1,  "lambda_sty": 1, "lambda_ds": 1, "lambda_reg": 1}, # weight_loss (eventually tweak lambda_ds (original authors set it to 1 for CelebaHQ and 2 for AFHQ))
#     {"generator": 0.999,"mapping_network": 0.999, "style_encoder": 0.999}, # beta_moving_avg (Looks 0.99 to 0.999 looks to have better behavior)
#     latent_dim)# latent_dim

# batch_size = 8 #len(dataset_fold[fold]["train"])

# # from lightning.pytorch.strategies import DDPStrategy

# trainer = L.Trainer(
#                     accelerator="gpu",
#                     devices=2,
#                     precision="bf16-mixed",
#                     #strategy="ddp_notebook",
#                     strategy="ddp_find_unused_parameters_true",#DDPStrategy(static_graph=True)
#                     max_epochs=max_epoch,
#                     logger=tb_logger,
#                     #num_sanity_val_steps=2, #to use only if you know the trainer is working !
#                     callbacks=[checkpoint_callback],
#                     #enable_checkpointing=False,
#                     enable_progress_bar=True,
#                     log_every_n_steps=1,
#                     deterministic=True
#                     #profiler="simple"
#                     )

# import resource
# soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

# trainer.fit(lit_model, train_dataloaders=[DataLoader(dataset_fold[fold]["train"], batch_size=batch_size,
#                                                      num_workers=1, persistent_workers=True,
#                                                      shuffle=True, drop_last=True),
#                                           DataLoader(dataset_fold_ref[fold]["train"], batch_size=batch_size,
#                                                      num_workers=1, persistent_workers=True,
#                                                      shuffle=True, drop_last=True)])



"""Generate fake image for each style transfer"""

"""Load trained StarGANv2 and trained Generator"""
"""CAREFUL THINK TO DENORMALIZE OUTPUT OF THE GENERATOR OR NORMALIZE INPUT IMAGE INTO THE GENERATOR"""

def generate_dataset(starganv2_path, fake_img_path_preffix, dataset_fold, mode="ref", fold=0, split="train",
                     batch_size=32, num_outs_per_domain=10, use_ema=True):
    # Load checkpoint
    StarGANv2_module = LightningStarGANV2.load_from_checkpoint(starganv2_path,
                                                               generator=conv_model.Generator,
                                                               mapping_network=conv_model.MappingNetwork,
                                                               style_encoder=conv_model.StyleEncoder,
                                                               discriminator=conv_model.Discriminator)

    if use_ema:
        StarGANv2_module.copy_parameters_from_weight(StarGANv2_module.generator, StarGANv2_module.generator_ema_weight)
        StarGANv2_module.copy_parameters_from_weight(StarGANv2_module.mapping_network, StarGANv2_module.mapping_network_ema_weight)
        StarGANv2_module.copy_parameters_from_weight(StarGANv2_module.style_encoder, StarGANv2_module.style_encoder_ema_weight)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = StarGANv2_module.generator.to(device)
    mapping_network = StarGANv2_module.mapping_network.to(device)
    style_encoder = StarGANv2_module.style_encoder.to(device)
    latent_dim = StarGANv2_module.latent_dim


    suffix = "_ema" if use_ema else ""
    fake_img_path = Path(fake_img_path_preffix + suffix)
    sub_directory = Path(split) / f"fold_{fold}" / mode / "imgs_labels_groups.zarr"

    store = zarr.DirectoryStore(fake_img_path / sub_directory)
    is_dataset_existing = False

    dataset = dataset_fold[fold][split]

    imgs_path, channel, fold_idx = dataset.imgs_path, dataset.channel, dataset.fold_idx
    domains = np.unique(dataset.imgs_zarr["labels"][fold_idx[:]])
    for i, cls_trg in enumerate(tqdm(domains, position=0, desc="y_trg")):
        mask_trg = dataset.imgs_zarr["labels"][fold_idx[:]] == cls_trg
        idx_trg = fold_idx[mask_trg]
        idx_org = fold_idx[~mask_trg]
        loader_org = DataLoader(custom_dataset.ImageDataset_all_info(imgs_path,
                                                                     channel=id_channel,
                                                                     fold_idx=idx_org,
                                                                     img_transform=v2.Compose([v2.Lambda(lambda img:
                                                                                                         torch.tensor(img, dtype=torch.float32)),
                                                                                               v2.Normalize(mean=len(channel)*[0.5],
                                                                                                            std=len(channel)*[0.5])]),
                                                                     label_transform=lambda label: torch.tensor(label, dtype=torch.long)),
                                batch_size=batch_size,
                                num_workers=1, persistent_workers=True)
        if mode == "ref":
            loader_ref = DataLoader(custom_dataset.ImageDataset(imgs_path,
                                                                channel=id_channel,
                                                                fold_idx=idx_trg,
                                                                img_transform=v2.Compose([v2.Lambda(lambda img:
                                                                                                    torch.tensor(img, dtype=torch.float32)),
                                                                                          v2.Normalize(mean=len(channel)*[0.5],
                                                                                                       std=len(channel)*[0.5])]),
                                                                label_transform=lambda label: torch.tensor(label, dtype=torch.long)),
                                    batch_size=batch_size,
                                    num_workers=1, persistent_workers=True,
                                    shuffle=True)
        for batch in tqdm(loader_org, position=0, desc="x_real_batch", leave=True):
            x_real, y_org, groups_org, indices_org = batch
            N = y_org.size(0)
            x_real, y_org = x_real.to(device), y_org.to(device)
            for i  in range(num_outs_per_domain):
                with torch.no_grad():
                    if mode == 'lat':
                        y_trg = torch.tensor([cls_trg] * N).to(device)
                        z_trg = torch.randn(N, latent_dim).to(device)
                        s_trg = mapping_network(z_trg, y_trg)
                    else:
                        try:
                            x_ref, y_trg = next(iter_ref)
                            x_ref, y_trg = x_ref.to(device), y_trg.to(device)
                            if y_trg.size(0) < N:
                                iter_ref = iter(loader_ref)
                                x_ref, y_trg = next(iter_ref)
                                x_ref, y_trg = x_ref.to(device), y_trg.to(device)
                        except:
                            iter_ref = iter(loader_ref)
                            x_ref, y_trg = next(iter_ref)
                            x_ref, y_trg = x_ref.to(device), y_trg.to(device)

                        if x_ref.size(0) > N:
                            x_ref = x_ref[:N]
                            y_trg = y_trg[:N]
                        s_trg = style_encoder(x_ref, y_trg)
                    x_fake = (generator(x_real, s_trg) + 1) / 2 # output of the generator are normalized. So we denormalize.
                    x_fake.clamp_(0, 1).unsqueeze_(1) # clip the value between 0 and 1 and stack generation round after batch dim
                    x_fake = x_fake.cpu()
                    if i == 0:
                        x_fake_stack = x_fake
                    else:
                        x_fake_stack = torch.cat([x_fake_stack, x_fake], dim=1)
            x_fake_stack = x_fake_stack.numpy()
            y_trg = y_trg.cpu().numpy()
            y_org = y_org.cpu().numpy()
            indices_org = indices_org.cpu().numpy()
            if not is_dataset_existing:
                xr.Dataset({"imgs":(["batch", "fake", "channel", "y", "x"], x_fake_stack),
                             "labels": ("batch", y_trg),
                             "labels_org": ("batch", y_org),
                             "groups_org": ("batch", np.array([*groups_org])),
                             "idx_org": ("batch", indices_org)}).to_zarr(
                                 store=store, mode="w",
                                 encoding={"imgs": {"chunks": (1, 1, 1, x_fake_stack.shape[3], x_fake_stack.shape[4])},
                                           "labels": {"chunks": 1},
                                           "labels_org": {"chunks": 1},
                                           "groups_org": {"chunks": 1},
                                           "idx_org": {"chunks": 1}})
                is_dataset_existing = True
            else:
                xr.Dataset({"imgs":(["batch", "fake", "channel", "y", "x"], x_fake_stack),
                            "labels": ("batch", y_trg),
                            "labels_org": ("batch", y_org),
                            "groups_org": ("batch", np.array([*groups_org])),
                            "idx_org": ("batch", indices_org)}).to_zarr(
                                store=store, append_dim="batch")


def plot_fake_img(fake_img_path_preffix, real_img_path, dataset_fold, mode="ref", fold=0, split="train", use_ema=True,
                  num_img_per_domain=2, seed=42):

    suffix = "_ema" if use_ema else ""
    fake_img_path = Path(fake_img_path_preffix + suffix)
    sub_directory = Path(split) / f"fold_{fold}" / mode / "imgs_labels_groups.zarr"
    imgs_zarr_fake = zarr.open(fake_img_path / sub_directory)
    fold_idx = dataset_fold[fold][split].fold_idx
    imgs_zarr =  zarr.open(Path(real_img_path))
    num_cols = imgs_zarr["imgs"][0].shape[0] + 1

    real_labels = imgs_zarr["labels"].oindex[fold_idx]
    domains = np.unique(real_labels)
    rng = np.random.default_rng(seed)
    fig, axs = plt.subplots(len(domains) * num_img_per_domain * (len(domains)-1), num_cols, figsize=(40, 70), squeeze=False)
    N = num_img_per_domain * (len(domains)-1)
    n = (len(domains)-1)
    for i, label in enumerate(domains):
        label_idx = fold_idx[real_labels == label]
        label_idx = rng.choice(label_idx, size=num_img_per_domain, replace=False)
        for j, idx in enumerate(label_idx):
            fake_idx = np.arange(len(imgs_zarr_fake["idx_org"]))[imgs_zarr_fake["idx_org"].oindex[:] == idx]
            for k, fk_idx in enumerate(fake_idx):
                fake_imgs = imgs_zarr_fake["imgs"].oindex[fk_idx]
                fake_label = imgs_zarr_fake["labels"].oindex[fk_idx]
                real_img = imgs_zarr["imgs"].oindex[idx, id_channel]
                axs[N * i + n * j + k][0].imshow(real_img.transpose(1, 2, 0))
                if k == 0:
                    axs[N * i + n * j + k][0].set_title(f"Real image - idx: {idx} - Class_{label}")
                for l in range(fake_imgs.shape[0]):
                    axs[N * i + n * j + k][l+1].imshow(fake_imgs[l].transpose(1, 2, 0))
                    if l == 0:
                        axs[N * i + n * j + k][l+1].set_title(f"Fake image - idx: {idx} - Class_{fake_label}")

    for ax in axs.flatten():
        ax.axis("off")

    fig.suptitle(f"Real vs Fake img - fold: {fold} - split: {split} - mode: {mode}", y=0.9)
    fig_name = f"real_fake_img_fold_{fold}_split_{split}_mode_{mode}" + suffix
    fig.savefig(fig_directory / fig_name)


starganv2_path = Path("lightning_checkpoint_log") / "StarGANv2_image_active_fold_0_epoch=29-step=70400.ckpt"
mode = "lat"#ref or lat
fold = 0
split = "test"
batch_size = 64
num_outs_per_domain = 3
fake_img_path_preffix = "image_active_dataset/fake_imgs"
use_ema = True
# generate_dataset(starganv2_path, fake_img_path_preffix, dataset_fold, mode=mode, fold=fold, split=split,
#                  batch_size=batch_size, num_outs_per_domain=num_outs_per_domain, use_ema=use_ema)


num_img_per_domain = 2
seed = 42
real_img_path = "image_active_dataset/imgs_labels_groups.zarr"
# plot_fake_img(fake_img_path_preffix, real_img_path,  dataset_fold, mode=mode, fold=fold,
#               split=split, use_ema=use_ema, num_img_per_domain=num_img_per_domain, seed=seed)


suffix = "_ema" if use_ema else ""
fake_img_path = Path(fake_img_path_preffix + suffix)
sub_directory = Path(split) / f"fold_{fold}" / mode / "imgs_labels_groups.zarr"
imgs_fake_path = fake_img_path / sub_directory

"""Confusion matrix of real image and fake_img"""
# # This path is for non-normalized images !
# # "VGG_image_active_fold_0epoch=41-train_acc=0.94-val_acc=0.92.ckpt"

# VGG_path = "VGG_image_active_fold_0epoch=78-train_acc=0.96-val_acc=0.91.ckpt"
# VGG_module = LightningModelV2.load_from_checkpoint(Path("lightning_checkpoint_log") / VGG_path,
#                                                    model=conv_model.VGG_ch)

# batch_size = 32
# fold = 0
# train_dataloaders = [DataLoader(dataset_fold[fold]["train"], batch_size=batch_size,
#                                 num_workers=1, persistent_workers=True,
#                                 shuffle=True),
#                      DataLoader(dataset_fold_ref[fold]["train"], batch_size=batch_size,
#                                 num_workers=1, persistent_workers=True,
#                                 shuffle=True)]

# val_dataloaders = [DataLoader(dataset_fold[fold]["val"], batch_size=batch_size,
#                               num_workers=1, persistent_workers=True,
#                               shuffle=True),
#                    DataLoader(dataset_fold_ref[fold]["val"], batch_size=batch_size,
#                               num_workers=1, persistent_workers=True,
#                               shuffle=True)]

# test_dataloaders = [DataLoader(dataset_fold[fold]["test"], batch_size=batch_size,
#                                num_workers=1, persistent_workers=True,
#                                shuffle=True),
#                     DataLoader(dataset_fold_ref[fold]["test"], batch_size=batch_size,
#                                num_workers=1, persistent_workers=True,
#                                shuffle=True)]

# dataset_fake = custom_dataset.ImageDataset_fake(imgs_fake_path,
#                                                 img_transform=v2.Compose([v2.Lambda(lambda img:
#                                                                                      torch.tensor(img, dtype=torch.float32)),
#                                                                           v2.Normalize(mean=len(channel)*[0.5],
#                                                                                        std=len(channel)*[0.5])]),
#                                                 label_transform=lambda label: torch.tensor(label, dtype=torch.long))
# fake_dataloader = DataLoader(dataset_fake, batch_size=batch_size, num_workers=1, persistent_workers=True)

# # Validation should be handled on a single devide (not ddp) Lightning Recommendation
# torch.set_float32_matmul_precision('medium') #try 'high')
# seed_everything(42, workers=True)
# tb_logger = pl_loggers.TensorBoardLogger(save_dir=Path("logs"), name="VGG_image_active_test")
# trainer = L.Trainer(accelerator="gpu",
#                     devices=1,
#                     logger=tb_logger,
#                     #precision="bf16-mixed",
#                     num_sanity_val_steps=0, #to use only if you know the trainer is working !
#                     enable_checkpointing=False,
#                     enable_progress_bar=True,
#                     )

# import resource
# soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))


# # trainer.test(VGG_module, train_dataloaders[0])
# # trainer.logger = pl_loggers.TensorBoardLogger(save_dir=Path("logs"), name="VGG_image_active_test")
# # trainer.test(VGG_module, val_dataloaders[0])
# # trainer.logger = pl_loggers.TensorBoardLogger(save_dir=Path("logs"), name="VGG_image_active_test")
# # trainer.test(VGG_module, test_dataloaders[0])
# # trainer.logger = pl_loggers.TensorBoardLogger(save_dir=Path("logs"), name="VGG_image_active_test")
# trainer.test(VGG_module, fake_dataloader)


# # imgs_zarr = zarr.open(imgs_fake_path)
# # length = imgs_zarr["imgs"].shape[0]
# # num_outs_per_domain = imgs_zarr["imgs"].shape[1]
# # indices = np.arange(length * num_outs_per_domain)[[3, 100, 20]]
# # true_idx, img_rank = np.divmod(indices, num_outs_per_domain)

# # imgs2 = np.stack(list(map(lambda x: imgs_zarr["imgs"][*x], zip(true_idx, img_rank))))

# # unique_true_idx, unique_img_rank = np.array(list(set(true_idx))), np.array(list(set(img_rank)))
# # map_true_idx, map_img_rank = {idx: i for i, idx in enumerate(unique_true_idx)}, {idx: i for i, idx in enumerate(unique_img_rank)}
# # new_true_idx, new_img_rank = [map_true_idx[idx] for idx in true_idx], [map_img_rank[idx] for idx in img_rank]
# # imgs1 = imgs_zarr["imgs"].oindex[unique_true_idx, unique_img_rank][new_true_idx, new_img_rank]

def compute_fid_lpips(fake_img_path_preffix,  dataset_fold, mode="ref",
                      fold=0, split="train", use_ema=True, batch_size=64):

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load fake dataset
    suffix = "_ema" if use_ema else ""
    fake_img_path = Path(fake_img_path_preffix + suffix)
    sub_directory = Path(split) / f"fold_{fold}" / mode / "imgs_labels_groups.zarr"
    imgs_fake_path = fake_img_path / sub_directory
    imgs_zarr_fake = zarr.open(imgs_fake_path)
    num_outs_per_domain = zarr.open(imgs_fake_path)["imgs"].shape[1]

    # load real dataset and fetch fold index and labels.
    dataset = dataset_fold[fold][split]
    real_img_path, channel, fold_idx = dataset.imgs_path, dataset.channel, dataset.fold_idx
    imgs_zarr_real =  zarr.open(Path(real_img_path))
    labels_real = imgs_zarr_real["labels"].oindex[fold_idx]

    # torchmetrics
    fid_metric = FrechetInceptionDistance(feature=2048, reset_real_features=True, normalize=True, input_img_size=tuple(dataset[0][0].shape[-3:])).to(device)
    # torchmetric recommendation mode
    fid_metric.set_dtype(torch.float64)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", reduction="mean", normalize=True).to(device)

    domains = np.unique(labels_real)
    fid_dict = {f"from_{label_org}_to_{label}": 0 for label_org in domains for label in domains if label != label_org}
    lpips_dict = {f"from_{label_org}_to_{label}": 0 for label_org in domains for label in domains if label != label_org}
    for label in tqdm(domains, position=0, desc="y_trg"):
        label_real_idx = fold_idx[labels_real == label]

        loader_real = DataLoader(custom_dataset.ImageDataset(real_img_path,
                                                             channel=channel,
                                                             fold_idx=label_real_idx,
                                                             img_transform=v2.Lambda(lambda img: torch.tensor(img, dtype=torch.float32)),
                                                             label_transform=lambda label: torch.tensor(label, dtype=torch.long)),
                                 batch_size=batch_size,
                                 num_workers=1, persistent_workers=True)
        # Allow to no reset real statistics as we compute for multiple pair
        # for instance we compare statistics compute on 0 with generated image (1 to 0, 2 to 0, 3 to 0)
        fid_metric.reset_real_features = False
        for batch in tqdm(loader_real, position=1, desc="fid_true", leave=True):
            X, y = batch
            X = X.to(device)
            fid_metric.update(X, real=True)
        del X
        torch.cuda.empty_cache()

        domains_org = [l for l in domains if l != label]
        for label_org in tqdm(domains_org, position=1, desc="y_org"):
            fake_idx = np.arange(len(imgs_zarr_fake["labels"]))[(imgs_zarr_fake["labels"].oindex[:] == label) &
                                                                (imgs_zarr_fake["labels_org"].oindex[:] == label_org)]
            loader_fake = DataLoader(custom_dataset.ImageDataset_fake(imgs_fake_path,
                                                                      mask_index=fake_idx,
                                                                      img_transform=v2.Lambda(lambda img: torch.tensor(img, dtype=torch.float32)),
                                                                      label_transform=lambda label: torch.tensor(label, dtype=torch.long)),
                                     batch_size=batch_size,
                                     num_workers=1, persistent_workers=True)
            # compute fid dist
            for batch in tqdm(loader_fake, position=1, desc="fid_false", leave=True):
                X, y = batch
                X = X.to(device)
                fid_metric.update(X, real=False)
            del X
            torch.cuda.empty_cache()

            fid_dict[f"from_{label_org}_to_{label}"] = fid_metric.compute().item()
            fid_metric.reset()

            # comput lpips score
            img_transform = v2.Lambda(lambda img: torch.tensor(img, dtype=torch.float32))
            batch_sampler_fake = BatchSampler(fake_idx, batch_size=batch_size//2, drop_last=False)
            for batch_idx in tqdm(batch_sampler_fake, position=1, desc="lpips", leave=True):
                # compute similarity between every pair of generated image from a same input but with different ref or lat code.
                for i in range(num_outs_per_domain-1):
                    imgs1 = img_transform(imgs_zarr_fake["imgs"].oindex[batch_idx, i]).to(device)
                    for j in range(i+1, num_outs_per_domain):
                        imgs2 = img_transform(imgs_zarr_fake["imgs"].oindex[batch_idx, j]).to(device)
                        lpips_metric.update(imgs1, imgs2)
            del imgs1, imgs2
            torch.cuda.empty_cache()

            lpips_dict[f"from_{label_org}_to_{label}"] = lpips_metric.compute().item()
            lpips_metric.reset()

        fid_metric.reset_real_features = True
        fid_metric.reset()

    return fid_dict, lpips_dict

# Because there is a large number of cpu available and that we are working with a relatively small number of samples, it leads to significant overhead
# when computing linalg.sqrtm within the fid metric. So need to turn down the number of thread to reduce this overhead.
os.environ["OMP_NUM_THREADS"] = "1"
fid_dict, lpips_dict = compute_fid_lpips(fake_img_path_preffix,  dataset_fold, mode=mode, fold=fold,
                                         split=split, use_ema=use_ema, batch_size=512)
