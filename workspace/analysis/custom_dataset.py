import numpy as np
from torch.utils.data import Dataset
import zarr
from itertools import groupby
from typing import Optional

class ImageDataset(Dataset):
    def __init__(self, imgs_path, channel, fold_idx, img_transform=None, label_transform=None, img_key="imgs", lbl_key="labels"):
        super().__init__()
        self.imgs_zarr = zarr.open(imgs_path)
        self.imgs_path = imgs_path
        self.channel = channel
        self.img_key = img_key
        self.lbl_key = lbl_key
        self.fold_idx = fold_idx
        if self.fold_idx is None:
            self.fold_idx = list(range(len(self.imgs_zarr[self.lbl_key])))
        self.img_transform = img_transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.fold_idx)

    def __getitem__(self, idx):
        imgs = self.imgs_zarr[self.img_key].oindex[self.fold_idx[idx], self.channel]
        labels = self.imgs_zarr[self.lbl_key].oindex[self.fold_idx[idx]]
        if self.img_transform is not None:
            imgs = self.img_transform(imgs)
        if self.label_transform is not None:
            labels = self.label_transform(labels)
        return imgs, labels

class ImageDataset_all_info(Dataset):
    def __init__(self, imgs_path, channel, fold_idx, img_transform=None, label_transform=None):
        super().__init__()
        self.imgs_zarr = zarr.open(imgs_path)
        self.imgs_path = imgs_path
        self.channel = channel
        self.fold_idx = fold_idx
        if self.fold_idx is None:
            self.fold_idx = list(range(len(self.imgs_zarr["labels"])))
        self.img_transform = img_transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.fold_idx)

    def __getitem__(self, idx):
        imgs = self.imgs_zarr["imgs"].oindex[self.fold_idx[idx], self.channel]
        labels = self.imgs_zarr["labels"].oindex[self.fold_idx[idx]]
        groups = self.imgs_zarr["groups"].oindex[self.fold_idx[idx]]
        indices = self.fold_idx[idx]
        if self.img_transform is not None:
            imgs = self.img_transform(imgs)
        if self.label_transform is not None:
            labels = self.label_transform(labels)
        return imgs, labels, groups, indices

class ImageDataset_Ref(Dataset):
    def __init__(self, imgs_path, channel, fold_idx, img_transform=None, label_transform=None, seed=42, img_key="imgs", lbl_key="labels"):
        super().__init__()
        self.imgs_zarr = zarr.open(imgs_path)
        self.imgs_path = imgs_path
        self.channel = channel
        self.img_key = img_key
        self.lbl_key = lbl_key
        self.fold_idx = fold_idx
        if self.fold_idx is None:
            self.fold_idx = list(range(len(self.imgs_zarr[self.lbl_key])))
        self.imgs_idx, self.imgs2_idx = self._make_dataset(seed)
        self.img_transform = img_transform
        self.label_transform = label_transform

    def _make_dataset(self, seed):
        labels = self.imgs_zarr[self.lbl_key].oindex[self.fold_idx]
        domain_idx = sorted(zip(self.fold_idx, labels), key=lambda x: x[1])
        domain_idx = {k: np.array(list(zip(*g))[0]) for k, g in groupby(domain_idx,  key=lambda x:x[1])}
        rng = np.random.default_rng(seed)
        return list(map(lambda list_idx: np.concatenate(list_idx),
                        list(zip(*[(group_idx,
                                    rng.choice(group_idx, size=len(group_idx), replace=False))
                                   for group_idx in list(domain_idx.values())]))))

    def __len__(self):
        return len(self.fold_idx)

    def __getitem__(self, idx):
        imgs = self.imgs_zarr[self.img_key].oindex[self.imgs_idx[idx], self.channel]
        imgs2 = self.imgs_zarr[self.img_key].oindex[self.imgs2_idx[idx], self.channel]
        labels = self.imgs_zarr[self.lbl_key].oindex[self.imgs_idx[idx]]
        if self.img_transform is not None:
            imgs = self.img_transform(imgs)
            imgs2 = self.img_transform(imgs2)
        if self.label_transform is not None:
            labels = self.label_transform(labels)
        return imgs, imgs2, labels


class ImageDataset_fake(Dataset):
    def __init__(self, imgs_path, mask_index=None, img_transform=None, label_transform=None):
        super().__init__()
        self.imgs_zarr = zarr.open(imgs_path)
        self.num_outs_per_domain = self.imgs_zarr["imgs"].shape[1]
        self.imgs_path = imgs_path
        self.img_transform = img_transform
        self.label_transform = label_transform
        if mask_index is None:
            self.indices_tot = np.arange(self.imgs_zarr["imgs"].shape[0] * self.num_outs_per_domain)
        else:
            list_idx = (mask_index * self.num_outs_per_domain).reshape(-1, 1)
            add_index = np.arange(self.num_outs_per_domain)
            self.indices_tot = (list_idx + add_index).reshape(-1)

    def __len__(self):
        return len(self.indices_tot)

    def __getitem__(self, idx):
        indices = self.indices_tot[idx]
        true_idx, img_rank = np.divmod(indices, self.num_outs_per_domain)
        if len(indices.shape) == 0 :
            imgs = self.imgs_zarr["imgs"].oindex[true_idx, img_rank]
        else:
            imgs = np.stack(self.imgs_zarr["imgs"][x, y] for x,y in zip(true_idx, img_rank))
        labels = self.imgs_zarr["labels"].oindex[true_idx]
        if self.img_transform is not None:
            imgs = self.img_transform(imgs)
        if self.label_transform is not None:
            labels = self.label_transform(labels)
        return imgs, labels


class ImageDataset_real_fake(Dataset):
    def __init__(self, imgs_real_path, imgs_fake_path,
                 channel, org_to_trg_label:Optional[list[tuple[int]]]=None,
                 img_transform=None, label_transform=None):
        super().__init__()
        self.imgs_real_path = imgs_real_path
        self.imgs_fake_path = imgs_fake_path
        self.imgs_zarr_real = zarr.open(imgs_real_path)
        self.imgs_zarr_fake = zarr.open(imgs_fake_path)
        self.channel = channel
        self.org_to_trg_label = org_to_trg_label
        self.img_transform = img_transform
        self.label_transform = label_transform

        if self.org_to_trg_label is not None:
            mask = np.sum([(self.imgs_zarr_fake["labels"].oindex[:] == label) &
                           (self.imgs_zarr_fake["labels_org"].oindex[:] == label_org)
                           for (label_org, label) in self.org_to_trg_label], axis=0)
            self.indices_tot = np.arange(self.imgs_zarr_fake["labels"].shape[0])[mask.astype(bool)]
        else:
            self.indices_tot = np.arange(self.imgs_zarr_fake["labels"].shape[0])


    def __len__(self):
        return len(self.indices_tot)

    def __getitem__(self, idx):
        indices = self.indices_tot[idx]
        # the first generated is chosen by default but it can be decided otherwise.
        imgs_fake = self.imgs_zarr_fake["imgs"].oindex[indices, 0]
        labels_fake = self.imgs_zarr_fake["labels"].oindex[indices]
        imgs_real = self.imgs_zarr_real["imgs"].oindex[self.imgs_zarr_fake["idx_org"].oindex[indices], self.channel]
        labels_real = self.imgs_zarr_real["labels"].oindex[self.imgs_zarr_fake["idx_org"].oindex[indices]]
        if self.img_transform is not None:
            imgs_fake = self.img_transform(imgs_fake)
            imgs_real = self.img_transform(imgs_real)
        if self.label_transform is not None:
            labels_fake = self.label_transform(labels_fake)
            labels_real = self.label_transform(labels_real)
        return imgs_real, imgs_fake, labels_real, labels_fake



"""
----------------- Only for profiles -----------------
"""
class RowDataset(Dataset):
    def __init__(self, row, labels, transform=None, target_transform=None):
        super().__init__()
        self.row = row
        self.row_labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.row_labels)

    def __getitem__(self, idx): 
        return self.row[idx,:], self.row_labels[idx]
