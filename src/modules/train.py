import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
from torch.amp import autocast, GradScaler

from src.data.utils import get_dataset, get_dataloader
from src.models.utils import get_model, ModelEMA
from src.modules.utils import (
    get_optimizer, get_scheduler, batch_to_device,
    flatten_dict, save_weights, run_eval
)
from src.logging.utils import get_logger

def train(cfg):
    logger = get_logger(cfg)

    df = pd.read_csv(cfg.train_csv_path)
    
    gkf = GroupKFold(n_splits=cfg.n_splits)
    splits = gkf.split(df, groups=df['SeriesInstanceUID'])
    
    train_idx, val_idx = next(s for i, s in enumerate(splits) if i == cfg.fold)
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    if cfg.fast_dev_run:
        train_df = train_df.head(cfg.batch_size * 2)
        val_df = val_df.head(cfg.batch_size_val * 2)

    train_ds = get_dataset(train_df, cfg, mode="train")
    train_dl = get_dataloader(train_ds, cfg, sampler=None, mode="train")

    val_ds = get_dataset(val_df, cfg, mode="val")
    val_dl = get_dataloader(val_ds, cfg, mode="val")

    model, _ = get_model(cfg)
    model.to(cfg.device)

    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg, n_steps=len(train_dl) * cfg.epochs)

    scaler = GradScaler() if cfg.mixed_precision else None
    ema_model = ModelEMA(model, decay=cfg.ema_decay) if cfg.ema else None

    best_score = 0
    for epoch in range(cfg.epochs):
        model.train()
        train_losses = []
        
        progress_bar = tqdm(train_dl, disable=cfg.local_rank != 0)
        for i, batch in enumerate(progress_bar):
            batch = batch_to_device(batch, device=cfg.device)

            with autocast(cfg.device.type, enabled=cfg.mixed_precision):
                output = model(batch)
                loss = output["loss"]
            
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()
            if ema_model:
                ema_model.update(model)

            train_losses.append(loss.item())

            if i % cfg.logging_steps == 0:
                train_metrics = {
                    "train": {"loss": np.mean(train_losses[-cfg.logging_steps:])},
                    "lr": scheduler.get_last_lr()[0], "epoch": epoch
                }
                progress_bar.set_postfix(flatten_dict(train_metrics))
                logger.log(train_metrics, commit=False)

        if (epoch + 1) % cfg.eval_epochs == 0:
            eval_model = ema_model.module if ema_model else model
            val_metrics = run_eval(eval_model, val_df, val_dl, cfg)

            current_score = val_metrics['val']['score']
            print(f"Epoch {epoch}: Val Score: {current_score:.4f}, Val Loss: {val_metrics['val']['loss']:.4f}")
            logger.log(val_metrics, commit=True)
            
            if current_score > best_score:
                print(f"Score improved: {best_score:.4f} -> {current_score:.4f}.")
                best_score = current_score
            print("Saving model...")
            save_weights(eval_model, cfg, epoch=f"{epoch}_score{current_score:.4f}")

    logger.finish()
    return