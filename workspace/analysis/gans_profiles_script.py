#!/usr/bin/env python
# coding: utf-8

# <center> <h1> GANs on cell profiles </center> </h1>
# 
# 
# # 1) Adaptation of naive_classifier notebook to get trained xgboost compatible with torch
# The goal of this Notebook is to adapt the [example of Diane](https://github.com/dlmbl/knowledge_extraction/blob/main/solution.ipynb) to create GANs but on cell profiles so 1D data instead of 2D. 



import pandas as pd

import numpy as np


from sklearn.preprocessing import RobustScaler


from data_split import StratifiedGroupKFold_custom

import torch
from torch.utils.data import DataLoader

import lightning as L
from lightning_parallel_training import LightningModelV2

from lightning.pytorch import seed_everything
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint
import conv_model

import custom_dataset

from pathlib import Path
import resource
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

#Loading of the data

metadata_df = pd.read_csv("target2_eq_moa2_active_metadata", index_col="ID")
features_df = pd.read_csv("target2_eq_moa2_active_features", index_col="ID")
features_df = features_df[features_df.columns[(features_df.isna().sum(axis=0) == 0) &
                                              ((features_df == np.inf).sum(axis=0) == 0) &
                                              (features_df.std() != 0)]]
#metadata_df = metadata_df.assign(moa_id=LabelEncoder().fit_transform(metadata_df["moa"]))
features_df = features_df.sort_index().reset_index(drop=True)
metadata_df = metadata_df.sort_index().reset_index()

kfold = list(StratifiedGroupKFold_custom().split(
    features_df, metadata_df["moa_id"], metadata_df["Metadata_InChIKey"]))

X = torch.tensor(features_df.values, dtype=torch.float)
y = torch.tensor(metadata_df["moa_id"].values, dtype=torch.long)

dataset_fold = {i:
                {"train": 0,
                 "test": 0} for i in range(len(kfold))}

for i in range(len(kfold)):
    scaler = RobustScaler()
    scaler.fit(X[kfold[i][0]])
    dataset_fold[i]["train"] = custom_dataset.RowDataset(torch.tensor(scaler.transform(X[kfold[i][0]]), dtype=torch.float), y[kfold[i][0]])
    dataset_fold[i]["test"] = custom_dataset.RowDataset(torch.tensor(scaler.transform(X[kfold[i][1]]), dtype=torch.float), y[kfold[i][1]])


# # 2 Deep Learning model
fold=0
# # Lightning Training
tb_logger = pl_loggers.TensorBoardLogger(save_dir=Path("logs"), name="SimpleNN_profiles_active")
checkpoint_callback = ModelCheckpoint(dirpath=Path("lightning_checkpoint_log"),
                                      filename=f"SimpleNN_profiles_active_fold_{fold}_RobustScaler_"+"{epoch}-{train_acc:.2f}-{val_acc:.2f}",
                                      save_top_k=1,
                                      monitor="val_acc",
                                      mode="max",
                                      every_n_epochs=2)

torch.set_float32_matmul_precision('medium') #try 'high')
seed_everything(42, workers=True)

max_epoch=10
lit_model = LightningModelV2(conv_model.SimpleNN,
                           model_param=(3645,#input_size
                                        4,#num_classes
                                        [2048, 2048],#hidden_layer_L
                                        [0.4, 0.2]#p_dopout_L
                                        ),
                           lr=1e-4,
                           weight_decay=1e-1,
                           max_epoch=max_epoch,
                           n_class=4,
                           apply_softmax=False)

trainer = L.Trainer(
                    accelerator="gpu",
                    devices=1,
                    max_epochs=max_epoch,
                    logger=tb_logger,
                    #num_sanity_val_steps=0, #to use only if you know the trainer is working !
                    callbacks=[checkpoint_callback],
                    #enable_checkpointing=False,
                    enable_progress_bar=False,
                    log_every_n_steps=1
                    )

trainer.fit(lit_model, DataLoader(dataset_fold[fold]["train"], batch_size=len(dataset_fold[fold]["train"]), num_workers=1, persistent_workers=True),
            DataLoader(dataset_fold[fold]["test"], batch_size=len(dataset_fold[fold]["test"]), num_workers=1, persistent_workers=True))
"""
# "SimpleNN_profiles_fold_0_RobustScaler_epoch=461-train_acc=0.83-val_acc=0.55.ckpt"
# "SimpleNN_profiles_fold_1_RobustScaler_epoch=377-train_acc=0.87-val_acc=0.45.ckpt"
# "SimpleNN_profiles_fold_2_RobustScaler_epoch=145-train_acc=0.72-val_acc=0.50.ckpt"
# "SimpleNN_profiles_fold_3_RobustScaler_epoch=341-train_acc=0.88-val_acc=0.41.ckpt"
# "SimpleNN_profiles_fold_4_RobustScaler_epoch=209-train_acc=0.79-val_acc=0.51.ckpt"
# trained_model_path = [
#     "SimpleNN_profiles_fold_0epoch=999-train_acc=0.77-val_acc=0.57.ckpt",
#     "SimpleNN_profiles_fold_1epoch=999-train_acc=0.79-val_acc=0.37.ckpt",
#     "SimpleNN_profiles_fold_2epoch=999-train_acc=0.77-val_acc=0.46.ckpt",
#     "SimpleNN_profiles_fold_3epoch=999-train_acc=0.81-val_acc=0.35.ckpt",
#     "SimpleNN_profiles_fold_4epoch=999-train_acc=0.79-val_acc=0.47.ckpt",
#     ]


# # GAN training

fold=0
# # Lightning Training
tb_logger = pl_loggers.TensorBoardLogger(save_dir=Path("logs"), name="SimpleNN_GAN_profiles")
checkpoint_callback = ModelCheckpoint(dirpath=Path("lightning_checkpoint_log"),
                                      filename=f"SimpleNN_GAN_profiles_fold_{fold}_RobustScaler_"+"{epoch}-{train_acc:.2f}-{val_acc:.2f}",
                                      #save_top_k=1,
                                      #monitor="val_acc",
                                      #mode="max",
                                      every_n_epochs=500)

torch.set_float32_matmul_precision('medium') #try 'high')
seed_everything(42, workers=True)

max_epoch = 2000
lit_model = LightningGAN(
    conv_model.VectorUNet, # inner_generator,
    conv_model.SimpleNN, # style_encoder,
    conv_model.VectorGenerator, # generator,
    conv_model.SimpleNN, # discriminator,
    {"input_dim":5150, "hidden_dims":[4096, 2048, 1024], "output_dim":3650}, # inner_generator_param,
    {"input_size":3650, "num_classes": 1500, "hidden_layer_L":[3000],
     "p_dopout_L":[0], "batchnorm":False}, # style_encoder_param,
    {"batchnorm_dim": 1500}, #generator_param
    {"input_size":3650, "num_classes": 7, "hidden_layer_L":[2048, 2048, 1024, 1024],
     "p_dopout_L":[0, 0, 0, 0], "batchnorm":False}, # discriminator_param,
    {"lr": 1e-4}, # adam_param_g,
    {"lr": 1e-5}, # adam_param_d,
    0.8, # beta_moving_avg,
    7, #n_class
    True, # if_reshape_vector=False,
    50, # H_target_shape=None
    False) # apply_softmax=False

batch_size = 293 #len(dataset_fold[fold]["train"])

trainer = L.Trainer(
                    accelerator="gpu",
                    devices=1,
                    precision=16,
                    #strategy="ddp_find_unused_parameters_true"
                    max_epochs=max_epoch,
                    logger=tb_logger,
                    #num_sanity_val_steps=0, #to use only if you know the trainer is working !
                    callbacks=[checkpoint_callback],
                    #enable_checkpointing=False,
                    enable_progress_bar=False,
                    log_every_n_steps=len(dataset_fold[fold]["train"]) // batch_size
                    #profiler="simple"
                    )

trainer.fit(lit_model, DataLoader(dataset_fold[fold]["train"], batch_size=batch_size,
                                  num_workers=1, persistent_workers=True))#,
                                  #shuffle=True))


## GAN Evaluation

# split = "train"
# num_images = 500
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# trained_generator = trainer.model.generator.to(device)
# trained_model_path = [
#     "SimpleNN_profiles_fold_0_RobustScaler_epoch=461-train_acc=0.83-val_acc=0.55.ckpt",
#     "SimpleNN_profiles_fold_1_RobustScaler_epoch=377-train_acc=0.87-val_acc=0.45.ckpt",
#     "SimpleNN_profiles_fold_2_RobustScaler_epoch=145-train_acc=0.72-val_acc=0.50.ckpt",
#     "SimpleNN_profiles_fold_3_RobustScaler_epoch=341-train_acc=0.88-val_acc=0.41.ckpt",
#     "SimpleNN_profiles_fold_4_RobustScaler_epoch=209-train_acc=0.79-val_acc=0.51.ckpt"
#                      ]

# trained_model = {i: LightningModel.load_from_checkpoint(Path("lightning_checkpoint_log") / trained_model_path[i],
#                                                      model=conv_model.SimpleNN).model.eval() #disable batchnorm and dropout
#               for i in list(dataset_fold.keys())}

# prototypes = {}
# for i in range(7):
#     options = np.where(dataset_fold[fold][split].row_labels == i)[0]
#     image_index = 0
#     x, y = dataset_fold[fold][split][options[image_index]]
#     prototypes[i] = x

# file_directory = Path("/home/hhakem/projects/counterfactuals_projects/workspace/analysis/figures")
# fig, axs = plt.subplots(1, 7, figsize=(12, 4))
# for i, ax in enumerate(axs):
#     ax.imshow(prototypes[i].reshape(1, 50, -1).permute(1, 2, 0))
#     ax.axis("off")
#     ax.set_title(f"Prototype {i}")
# fig.savefig(file_directory / f"{split}_profiles_styles_fold_{fold}.png")
# plt.close(fig)


# random_test = torch.utils.data.Subset(
#     dataset_fold[fold][split], np.random.choice(len(dataset_fold[fold][split]), num_images, replace=False)
# )
# counterfactuals = np.zeros((7, num_images, *prototypes[0].shape))

# predictions = []
# source_labels = []
# target_labels = []
# with torch.inference_mode():
#     trained_generator.eval()
#     for i, (x, y) in tqdm(enumerate(random_test), total=num_images, leave=True):
#         for lbl in range(7):
#             # Create the counterfactual
#             x_fake = trained_generator(
#                 x.unsqueeze(0).to(device), prototypes[lbl].unsqueeze(0).to(device)
#             )
#             # Predict the class of the counterfactual image
#             pred = trained_model[fold](x_fake)

#             # Store the source and target labels
#             source_labels.append(y)  # The original label of the image
#             target_labels.append(lbl)  # The desired label of the counterfactual image
#             # Store the counterfactual image and prediction
#             counterfactuals[lbl][i] = x_fake.cpu().detach().numpy()
#             predictions.append(pred.argmax().item())

# fig, ax = plt.subplots(1,1, figsize=(12, 15))
# cf_cm = confusion_matrix(target_labels, predictions, normalize="true")
# sns.heatmap(cf_cm, annot=True, fmt=".2f", ax=ax)
# ax.set_ylabel("True")
# ax.set_xlabel("Predicted")
# fig.savefig(file_directory / "Condusion_matrix_generated_image")
# plt.close(fig)

# print(accuracy_score(target_labels, predictions))
"""
