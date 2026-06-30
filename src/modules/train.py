import json
from pathlib import Path

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

    start_epoch = 0
    if cfg.resume_from_checkpoint:
        print(f"--- Resuming training from checkpoint: {cfg.resume_from_checkpoint} ---")
        checkpoint = torch.load(cfg.resume_from_checkpoint, map_location=cfg.device)

        model.load_state_dict(checkpoint) 
        if ema_model:
            ema_model.module.load_state_dict(checkpoint)

        if cfg.resume_epoch:
            start_epoch = cfg.resume_epoch
            print(f"Starting from epoch {start_epoch}")
        
        if scheduler:
            scheduler.last_epoch = start_epoch * len(train_dl)
            print(f"Scheduler fast-forwarded to step {scheduler.last_epoch}.")

    best_score = 0
    if cfg.resume_from_checkpoint and hasattr(cfg, 'resume_best_score'):
        best_score = cfg.resume_best_score
        print(f"Resuming with best_score set to {best_score:.4f}")
    for epoch in range(start_epoch, cfg.epochs):
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


# =============================================================================
# Cell-tracking training entry point
# =============================================================================

def _save_celltrack(model, cfg, epoch: int, score: float):
    """Save model weights to cfg.save_dir (default: ./checkpoints/celltrack/)."""
    import pickle
    save_dir = Path(getattr(cfg, "save_dir", "checkpoints/celltrack"))
    save_dir.mkdir(parents=True, exist_ok=True)

    state = model.module.state_dict() if cfg.world_size > 1 else model.state_dict()
    stem = f"celltrack_epoch{epoch}_score{score:.4f}"
    pt_path = save_dir / f"{stem}.pt"
    torch.save(state, pt_path)
    print(f"SAVED WEIGHTS: {pt_path}")

    cfg_path = save_dir / f"{stem}.pkl"
    with open(cfg_path, "wb") as f:
        pickle.dump(cfg, f)


def run_eval_celltrack(model, val_dl, cfg) -> dict:
    """Evaluate detection + edge metrics on the validation set."""
    model.eval()
    device = cfg.device
    amp_dtype = torch.bfloat16 if cfg.mixed_precision else torch.float32

    det_losses, edge_losses = [], []
    edge_correct_total, edge_total_total = 0, 0

    with torch.no_grad():
        for batch in tqdm(val_dl, disable=cfg.local_rank != 0, desc="val"):
            batch = batch_to_device(batch, device=device)

            with autocast(device.type, enabled=cfg.mixed_precision, dtype=amp_dtype):
                output = model(batch)

            det_losses.append(output["det_loss"].item())
            edge_losses.append(output["edge_loss"].item())

            # Edge accuracy on annotated rows/cols
            masks = output["masks"]           # (B, W, M)
            edge_targets = output["edge_targets"]  # (B, W-1, M, M)
            logits_edge = output["logits_edge"]    # list of (B, M, M)

            W_minus1 = edge_targets.shape[1]
            for i in range(W_minus1):
                logits_i = logits_edge[i]     # (B, M, M)
                target_i = edge_targets[:, i] # (B, M, M)
                mask_t  = masks[:, i]         # (B, M)
                mask_t1 = masks[:, i + 1]     # (B, M)

                B = logits_i.shape[0]
                for b in range(B):
                    nt  = int(mask_t[b].sum().item())
                    nt1 = int(mask_t1[b].sum().item())
                    if nt == 0 or nt1 == 0:
                        continue
                    lg = logits_i[b, :nt, :nt1]    # (nt, nt1)
                    tgt = target_i[b, :nt, :nt1]   # (nt, nt1)
                    probs = torch.softmax(lg, dim=0)
                    preds = (probs > 0.5).float()
                    active_rows = tgt.sum(dim=1) > 0
                    active_cols = tgt.sum(dim=0) > 0
                    active_mask = active_rows.unsqueeze(1) | active_cols.unsqueeze(0)
                    if active_mask.any():
                        edge_correct_total += int((preds[active_mask] == tgt[active_mask]).sum().item())
                        edge_total_total += int(active_mask.sum().item())

    val_det  = float(np.mean(det_losses))  if det_losses  else 0.0
    val_edge = float(np.mean(edge_losses)) if edge_losses else 0.0
    edge_acc = edge_correct_total / max(edge_total_total, 1)

    # Score: edge accuracy (higher is better)
    return {
        "val": {
            "loss": val_det + val_edge,
            "det_loss": val_det,
            "edge_loss": val_edge,
            "edge_acc": edge_acc,
            "score": edge_acc,
        }
    }


def train_celltrack(cfg):
    """Training loop for cell tracking — mirrors train() structure."""
    logger = get_logger(cfg)

    # --- Load dataset paths from JSON splits file ---
    splits_path = Path(cfg.splits_json)
    if not splits_path.exists():
        raise FileNotFoundError(
            f"Splits file not found: {splits_path}\n"
            "Create a JSON file with keys 'train' and 'val' each containing "
            "a list of dataset paths (without extension)."
        )
    with open(splits_path) as f:
        splits = json.load(f)

    train_paths = splits.get("train", [])
    val_paths = splits.get("val", [])

    if cfg.fast_dev_run:
        train_paths = train_paths[:2]
        val_paths = val_paths[:1]

    train_df = pd.DataFrame({"ds_path": train_paths})
    val_df = pd.DataFrame({"ds_path": val_paths})

    # Build train dataset first to discover max_nodes, then share with val
    train_ds = get_dataset(train_df, cfg, mode="train")
    cfg.max_nodes = train_ds.max_nodes  # pin before building val

    train_dl = get_dataloader(train_ds, cfg, sampler=None, mode="train")

    val_ds = get_dataset(val_df, cfg, mode="val")
    val_dl = get_dataloader(val_ds, cfg, mode="val")

    if cfg.local_rank == 0:
        print(f"Training windows: {len(train_ds)}, Validation windows: {len(val_ds)}")
        print(f"Max nodes per frame: {cfg.max_nodes}")

    model, _ = get_model(cfg)
    model.to(cfg.device)

    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg, n_steps=len(train_dl) * cfg.epochs)

    amp_dtype = torch.bfloat16 if cfg.mixed_precision else torch.float32
    scaler = GradScaler() if (cfg.mixed_precision and amp_dtype == torch.float16) else None
    ema_model = ModelEMA(model, decay=cfg.ema_decay) if cfg.ema else None

    start_epoch = 0
    if cfg.resume_from_checkpoint:
        print(f"--- Resuming from checkpoint: {cfg.resume_from_checkpoint} ---")
        checkpoint = torch.load(cfg.resume_from_checkpoint, map_location=cfg.device)
        model.load_state_dict(checkpoint)
        if ema_model:
            ema_model.module.load_state_dict(checkpoint)
        if cfg.resume_epoch:
            start_epoch = cfg.resume_epoch
        if scheduler and cfg.resume_epoch:
            scheduler.last_epoch = start_epoch * len(train_dl)

    best_score = getattr(cfg, "resume_best_score", None) or 0.0

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        train_losses = {"loss": [], "det_loss": [], "edge_loss": []}

        progress_bar = tqdm(train_dl, disable=cfg.local_rank != 0)
        for i, batch in enumerate(progress_bar):
            batch = batch_to_device(batch, device=cfg.device)

            with autocast(cfg.device.type, enabled=cfg.mixed_precision, dtype=amp_dtype):
                output = model(batch)
                loss = output["loss"]

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

            optimizer.zero_grad()
            if scheduler:
                scheduler.step()
            if ema_model:
                ema_model.update(model)

            for k in train_losses:
                if k in output:
                    train_losses[k].append(output[k].item() if torch.is_tensor(output[k]) else output[k])

            if i % cfg.logging_steps == 0:
                recent = {k: float(np.mean(v[-cfg.logging_steps:])) for k, v in train_losses.items() if v}
                lr_now = scheduler.get_last_lr()[0] if scheduler else cfg.lr
                metrics = {"train": recent, "lr": lr_now, "epoch": epoch}
                progress_bar.set_postfix(flatten_dict(metrics))
                logger.log(metrics, commit=False)

        if (epoch + 1) % cfg.eval_epochs == 0:
            eval_model = ema_model.module if ema_model else model
            val_metrics = run_eval_celltrack(eval_model, val_dl, cfg)

            current_score = val_metrics["val"]["score"]
            print(
                f"Epoch {epoch}: "
                f"det_loss={val_metrics['val']['det_loss']:.4f}  "
                f"edge_loss={val_metrics['val']['edge_loss']:.4f}  "
                f"score={current_score:.4f}"
            )
            logger.log(val_metrics, commit=True)

            if current_score > best_score:
                print(f"Score improved: {best_score:.4f} → {current_score:.4f}")
                best_score = current_score

            _save_celltrack(eval_model, cfg, epoch, current_score)

    logger.finish()