import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import GroupKFold

import torch
from torch.amp import autocast, GradScaler
from monai.inferers import sliding_window_inference

from src.data.utils import get_dataset, get_dataloader
from src.models.utils import get_model, ModelEMA
from src.modules.utils import (
    get_optimizer, get_scheduler, batch_to_device,
    flatten_dict, save_weights
)
from src.logging.utils import get_logger
from src.modules.metric import score, LABEL_COLS 

def run_eval(model, val_ds, val_dl, cfg):
    model.eval()
    
    progress_bar = tqdm(range(len(val_dl)), disable=cfg.local_rank != 0)
    val_itr = iter(val_dl)
    
    all_preds = []
    all_labels = []
    val_losses = []

    with torch.no_grad():
        for i in progress_bar:
            batch = next(val_itr)
            batch = batch_to_device(batch, cfg.device)
            
            with autocast(cfg.device.type):
                preds_map = sliding_window_inference(
                    inputs=batch["input"].float(),
                    roi_size=cfg.roi_size,
                    predictor=model,
                    overlap=0.5,
                    sw_batch_size=4,
                )

                loss = model.loss_fn(preds_map, batch["target"].float()).item()
                val_losses.append(loss)

            preds_probs = torch.sigmoid(preds_map)
            location_probs = torch.max(preds_probs.view(preds_probs.shape[0], 13, -1), dim=2).values
            present_prob = torch.max(location_probs, dim=1, keepdim=True).values
            final_probs = torch.cat([location_probs, present_prob], dim=1)
            all_preds.append(final_probs.cpu().numpy())

            true_locations = torch.max(batch["target"].view(batch["target"].shape[0], 13, -1), dim=2).values
            true_presence = torch.max(true_locations, dim=1, keepdim=True).values
            final_labels = torch.cat([true_locations, true_presence], dim=1)
            all_labels.append(final_labels.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    val_metrics = score(y_true, y_pred)
    val_metrics['loss'] = np.mean(val_losses)
    
    return {"val": val_metrics}

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
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
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

        eval_model = ema_model.module if ema_model else model
        val_metrics = run_eval(eval_model, val_ds, val_dl, cfg)
        
        current_score = val_metrics['val']['score']
        print(f"Epoch {epoch}: Val Score: {current_score:.4f}, Val Loss: {val_metrics['val']['loss']:.4f}")
        logger.log(val_metrics, commit=True)
        
        if current_score > best_score:
            print(f"Score improved: {best_score:.4f} -> {current_score:.4f}. Saving model...")
            best_score = current_score
            save_weights(eval_model, cfg, epoch=f"{epoch}_score{current_score:.4f}")

    logger.finish()
    return