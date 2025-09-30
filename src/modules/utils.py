from copy import deepcopy
import pickle

import torch
import torch.nn as nn

def batch_to_device(batch, device):
    if isinstance(batch, dict):
        return {key: batch_to_device(val, device) for key, val in batch.items()}
    elif isinstance(batch, list):
        return [batch_to_device(val, device) for val in batch]
    else:
        return batch.to(device)

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

def save_weights(model, cfg, epoch=""):
    if epoch != "":
        epoch = f"_epoch{epoch}"

    if cfg.world_size > 1:
        state_dict= model.module.state_dict()
    else:
        state_dict= model.state_dict()

    # Weights
    fpath= "./data/checkpoints/{}_{}{}.pt".format(
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