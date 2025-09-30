from types import SimpleNamespace

from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseModel(nn.Module):
    def __init__(self, cfg: SimpleNamespace, inference_mode: bool = False):
        super().__init__()
        self.cfg = cfg
        self.inference_mode = inference_mode
        self.loss_fn = self._init_loss_fn()

    def _init_loss_fn(self):
        if self.inference_mode:
            return None
        mname, cname = self.cfg.loss_type.rsplit(".", 1)
        losses = import_module(mname)
        loss_class = getattr(losses, cname)
        return loss_class(**vars(self.cfg.loss_cfg))