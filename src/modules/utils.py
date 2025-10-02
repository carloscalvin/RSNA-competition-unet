from monai.inferers import sliding_window_inference
from src.modules.metric import score, LABEL_COLS
import pickle
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.amp import autocast

def batch_to_device(batch, device):
    if isinstance(batch, dict):
        return {key: batch_to_device(val, device) for key, val in batch.items()}
    elif isinstance(batch, list):
        return [batch_to_device(val, device) for val in batch]
    else:
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        return batch

def calc_grad_norm(parameters,norm_type=2.):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    if torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        total_norm = None
        
    return total_norm

def get_optimizer(model, cfg):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    return optimizer

def get_scheduler(optimizer, cfg, n_steps):
    if cfg.scheduler == "Constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    elif cfg.scheduler == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max = n_steps,
            eta_min = cfg.lr_min,
            )
    else:
        raise ValueError(f"{cfg.scheduler} is not a valid scheduler.")

def flatten_dict(d):
    def _flatten(current_key, nested_dict, flattened_dict):
        for k, v in nested_dict.items():
            new_key = f"{current_key}.{k}" if current_key else k
            if isinstance(v, dict) and v:
                _flatten(new_key, v, flattened_dict)
            elif v is not None and v != {}:  # Exclude None values and empty dictionaries
                flattened_dict[new_key] = v
    
    flattened_dict = {}
    _flatten("", d, flattened_dict)
    return flattened_dict

def run_eval(model, val_df, val_dl, cfg):
    model.eval()

    progress_bar = tqdm(range(len(val_dl)), disable=cfg.local_rank != 0)
    val_itr = iter(val_dl)
    
    all_preds = []
    all_series_uids = []
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

                #loss = model.loss_fn(preds_map, batch["target"].float()).item()
                #val_losses.append(loss)

            preds_probs = torch.sigmoid(preds_map)
            location_probs = torch.max(preds_probs.view(preds_probs.shape[0], 13, -1), dim=2).values
            present_prob = torch.max(location_probs, dim=1, keepdim=True).values
            final_probs = torch.cat([location_probs, present_prob], dim=1)
            
            all_preds.append(final_probs.cpu().numpy())
            all_series_uids.extend(batch["series_uid"])

    y_pred = np.concatenate(all_preds)
    ordered_df = val_df[val_df['SeriesInstanceUID'].isin(all_series_uids)].set_index('SeriesInstanceUID').loc[all_series_uids].reset_index()
    y_true = ordered_df[LABEL_COLS].values

    val_metrics = score(y_true, y_pred)
    val_metrics['loss'] = np.mean(val_losses)
    
    return {"val": val_metrics}

def save_weights(model, cfg, epoch=""):
    if epoch != "":
        epoch = f"_epoch{epoch}"

    if cfg.world_size > 1:
        state_dict= model.module.state_dict()
    else:
        state_dict= model.state_dict()

    # Weights
    fpath= "/content/drive/MyDrive/RSNA/models/RSNA-competition-unet/fold0/{}_{}{}.pt".format(
        cfg.config_file, 
        cfg.seed, 
        epoch,
        )
    torch.save(state_dict, fpath)
    print("SAVED WEIGHTS: ", fpath)
    
    # Config
    fpath= fpath.replace(".pt", ".pkl")
    with open(fpath, "wb") as f:
        pickle.dump(cfg, f)
    print("SAVED CFG: ", fpath)
    return