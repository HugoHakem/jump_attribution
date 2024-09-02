#!/usr/bin/env python
# coding: utf-8

# <center> <h1> GANs on cell profiles </center> </h1>
# 
# 
# # 1) Adaptation of naive_classifier notebook to get trained xgboost compatible with torch
# The goal of this Notebook is to adapt the [example of Diane](https://github.com/dlmbl/knowledge_extraction/blob/main/solution.ipynb) to create GANs but on cell profiles so 1D data instead of 2D. 




import polars as pl
import pandas as pd

import numpy as np
import cupy as cp

from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from xgboost import XGBClassifier

from features_engineering import features_drop_corr
from features_engineering import features_drop_corr_gpu

from data_split import StratifiedGroupKFold_custom



import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

import lightning as L
from lightning_parallel_training import LightningModel
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint
import conv_model

import custom_dataset
import captum.attr
from captum.attr import IntegratedGradients

from pathlib import Path
import resource
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

#Loading of the data

metadata_df = pd.read_csv("target2_eq_moa2_metadata", index_col="ID")
features_df = pd.read_csv("target2_eq_moa2_features", index_col="ID")
nan_col = features_df.columns[features_df.isna().sum(axis=0) > 0]
nan_col, len(nan_col)
inf_col = features_df.columns[(features_df == np.inf).sum(axis=0) > 0]
inf_col, len(inf_col)
features_df = features_df[features_df.columns[(features_df.isna().sum(axis=0) == 0) & 
                                              ((features_df == np.inf).sum(axis=0) == 0) &
                                              (features_df.std() != 0)]]
metadata_df = metadata_df.assign(moa_id=LabelEncoder().fit_transform(metadata_df["moa"]))
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
    dataset_fold[i]["train"] =  custom_dataset.RowDataset(torch.tensor(scaler.transform(X[kfold[i][0]]), dtype=torch.float), y[kfold[i][0]])
    dataset_fold[i]["test"] = custom_dataset.RowDataset(torch.tensor(scaler.transform(X[kfold[i][1]]), dtype=torch.float), y[kfold[i][1]])


# # 2 Deep Learning model
fold=4
# # Lightning Training
tb_logger = pl_loggers.TensorBoardLogger(save_dir=Path("logs"), name="SimpleNN_profiles")
checkpoint_callback = ModelCheckpoint(dirpath=Path("lightning_checkpoint_log"),
                                      filename=f"SimpleNN_profiles_fold_{fold}_RobustScaler_"+"{epoch}-{train_acc:.2f}-{val_acc:.2f}",
                                      save_top_k=1,
                                      monitor="val_acc",
                                      mode="max",
                                      every_n_epochs=2)

torch.set_float32_matmul_precision('medium') #try 'high')
seed_everything(42, workers=True)

max_epoch=1000
lit_model = LightningModel(conv_model.SimpleNN,
                           model_param=(3650,#input_size
                                        7,#num_classes
                                        [2048, 2048],#hidden_layer_L
                                        [0.4, 0.2]#p_dopout_L
                                        ),
                           lr=1e-4,
                           weight_decay=1e-1,
                           max_epoch=max_epoch,
                           n_class=7,
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
# fold_model = {i: LightningModel.load_from_checkpoint(Path("lightning_checkpoint_log") / trained_model_path[i],model=conv_model.SimpleNN).model
#               for i in list(dataset_fold.keys())}
# # 2) Get an attribution
# Goal understand what the model has learnt



# fold = 0
# model = trained_model[fold].to(device)

# batch_size = 7 #num class as well
# batch = []
# for i in range(batch_size):
#     batch.append(next(row for row in dataset_fold[fold]["train"] if row[1] == i))
# X = torch.stack([b[0] for b in batch])
# y = torch.tensor([b[1] for b in batch])
# X = X.to(device).requires_grad_()
# y = y.to(device)





# torch.gradient(model(X)[:, 0], spacing = X)





# model(X)[:, 0]





# X.shape