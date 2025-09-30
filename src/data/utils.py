from importlib import import_module

import torch
import numpy as np

def get_dataset(df, cfg, mode='train'):
    if mode == 'train':
        dataset = get_train_dataset(df, cfg)
    elif mode == 'val':
        dataset = get_val_dataset(df, cfg)
    else:
        pass
    return dataset

def get_dataloader(ds, cfg, sampler=None, mode='train'):
    if mode == 'train':
        dl = get_train_dataloader(ds, cfg, sampler=sampler)
    elif mode =='val':
        dl = get_val_dataloader(ds, cfg)
    return dl

def get_train_dataset(df, cfg):
    dpath= f"src.data.{cfg.dataset_type}"
    ds = import_module(dpath).CustomDataset(cfg=cfg, df=df, mode="train")

    if cfg.fast_dev_run:
        ds= torch.utils.data.Subset(ds, np.arange(cfg.batch_size))

    return ds

def get_val_dataset(df, cfg):
    dpath= f"src.data.{cfg.dataset_type}"
    ds = import_module(dpath).CustomDataset(cfg=cfg, df=df, mode="val")
    
    if cfg.fast_dev_run:
        ds= torch.utils.data.Subset(ds, np.arange(cfg.batch_size))
    return ds

def get_train_dataloader(ds, cfg, sampler):
    train_dataloader = torch.utils.data.DataLoader(
        ds,
        sampler= sampler,
        shuffle= True if sampler is None else False,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last,
        collate_fn= ds.collate_fn if hasattr(ds, "collate_fn") else None,
        timeout= 30,
        prefetch_factor=3,
    )
    if cfg.local_rank == 0:
        print(f"TRAIN: dataset {len(ds)} dataloader {len(train_dataloader)}")
    return train_dataloader

def get_val_dataloader(ds, cfg):
    sampler = torch.utils.data.SequentialSampler(ds)

    # Optional: different batch_size than train
    if cfg.batch_size_val is not None:
        batch_size = cfg.batch_size_val
    else:
        batch_size = cfg.batch_size

    val_dataloader = torch.utils.data.DataLoader(
        ds,
        sampler=sampler,
        batch_size=batch_size,
        num_workers= 4,
        pin_memory=cfg.pin_memory,
        collate_fn= ds.collate_fn if hasattr(ds, "collate_fn") else None,
        timeout= 30,
    )
    if cfg.local_rank == 0:
        print(f"VALID: dataset {len(ds)}, dataloader {len(val_dataloader)}")
    return val_dataloader

