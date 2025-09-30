from importlib import import_module
from copy import deepcopy

import torch
import torch.nn as nn

from .layers import count_parameters

def get_model(cfg, inference_mode: bool = False):
    # Build
    m = import_module(f"src.models.{cfg.model_type}").Net(
        cfg=cfg, 
        inference_mode=inference_mode,
        )

    # Param count
    n_params= count_parameters(m)
    if cfg.local_rank == 0:
        print(f"Model: {cfg.model_type}")
        print("n_param: {:_}".format(n_params))

    # Load weights
    f= cfg.weights_path
    if f != "":
        m.load_state_dict(torch.load(f, map_location=cfg.device, weights_only=True))
        if cfg.local_rank == 0:
            print("LOADED WEIGHTS:", f)

    return m, n_params

class ModelEMA(nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super().__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)