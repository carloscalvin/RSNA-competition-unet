from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from src.augs import aug3d, Mixup, CutmixSimple

from .layers import ResnetEncoder3d, UnetDecoder3d, SegmentationHead3d
from ._base import BaseModel


class Net(BaseModel):
    def __init__(
        self, 
        cfg: SimpleNamespace, 
        inference_mode: bool = False,
        ):
        super().__init__(cfg=cfg, inference_mode=inference_mode)
        self.cfg = cfg
        self.inference_mode = inference_mode

        # Last channels (~1.25x speedup on Ampere GPUs)
        if not inference_mode and \
           torch.cuda.is_available() and \
           torch.cuda.get_device_properties(0).major >= 8:
            self.last_channels= True
        else:
            self.last_channels= False
        
        # Augs
        self.mixup = Mixup(cfg.mixup_beta)
        self.cutmix = CutmixSimple()
        
        # Encoder
        self.backbone = ResnetEncoder3d(
            cfg= cfg,
            inference_mode= inference_mode,
            **vars(cfg.encoder_cfg),
            )
        ecs= self.backbone.channels[::-1]

        if self.last_channels:
            self.backbone = self.backbone.to(memory_format=torch.channels_last_3d)

        # Decoder + Heads
        self.decoder= UnetDecoder3d(
            encoder_channels= ecs,
            **vars(cfg.decoder_cfg),
        )

        self.seg_head= SegmentationHead3d(
            in_channels= self.decoder.decoder_channels[-1],
            out_channels= cfg.seg_classes,
        )

        if cfg.deep_supervision:
            self.aux_head= SegmentationHead3d(
                in_channels= ecs[0],
                out_channels= cfg.seg_classes,
            )

    def proc_flip(self, x_in, dim):
        # Flip TTA
        i = torch.flip(x_in, dim)
        f = self.backbone.forward_features(i)
        f = f[::-1]
        f = f[:len(self.cfg.decoder_cfg.decoder_channels)+1]
        p = self.seg_head(self.decoder(f)[-1])
        return torch.flip(p, dim)

    def forward(self, batch):
        
        # Augs
        if self.training:
            x= batch["input"].float() # bs,c,t,h,w
            x = x / 255.0
            y= batch["target"].float()

            # Cutmix
            if torch.rand(1)[0] < self.cfg.cutmix_p:
                x, y = self.cutmix(x, y)

            x, y= aug3d.coarse_dropout_3d(x, y, p=0.5)
            x, y= aug3d.rotate(x, y, p= 1.0, dims=[(-2,-1)])
            x, y= aug3d.flip_3d(x, y)
            x, y= aug3d.swap_dims(x, y, dims=(-2,-1))

            # Mixup
            if torch.rand(1)[0] < self.cfg.mixup_p:
                x, y = self.mixup(x, y)
    
        else:
            x= batch.float()
            x = x / 255.0

        if self.last_channels:
            x = x.to(memory_format=torch.channels_last_3d)

        # Forward pass
        x_in = x
        x_feats = self.backbone.forward_features(x)
        x = x_feats[::-1]
        x = x[:len(self.cfg.decoder_cfg.decoder_channels)+1] # remove unused feature maps 
        x= self.decoder(x)
        x_seg= self.seg_head(x[-1])

        if self.training:

            # Loss
            loss= self.loss_fn(x_seg, y)

            # Aux loss (max pixel vs max label)
            x_aux= F.max_pool3d(x_seg, kernel_size=4, stride=4)
            y_aux= F.max_pool3d(y, kernel_size=4, stride=4)
            loss_aux= self.loss_fn(x_aux, y_aux)
            loss += 0.25 * loss_aux

            if self.cfg.deep_supervision:

                # Downsample ys (way faster than upsample)
                x_aux= self.aux_head(x[-2])
                y_aux= F.avg_pool3d(y, kernel_size=2)
                loss_aux= self.loss_fn(x_aux, y_aux)
                loss += 0.1 * loss_aux

            return {
                "logits": x_seg,
                "loss": loss,
            }
        else:

            # TTA during inference
            if self.cfg.tta:
                p1 = self.proc_flip(x_in, [2])
                p2 = self.proc_flip(x_in, [3])
                x_seg = torch.mean(torch.stack([x_seg, p1, p2]), dim=0)
   
            else:
                pass
            
            return x_seg

if __name__ == "__main__":
    from src.configs.r3d200 import cfg
    from src.models.layers import count_parameters

    m= Net(cfg=cfg)#.cuda()
    m= m.eval()
    print("n_param: {:_}".format(count_parameters(m)))