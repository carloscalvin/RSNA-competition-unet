from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import DropPath
from timm.models._manipulate import checkpoint 

from .utils import load_weights

def conv3x3x3(ic, oc, stride=1):
    return nn.Conv3d(
        ic,
        oc,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
        )

class BasicBlock(nn.Module):
    def __init__(
        self, 
        ic, 
        oc, 
        stride: int = 1, 
        downsample: bool = None, 
        expansion_factor: int = 1,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module = nn.BatchNorm3d,
        act_layer: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.conv1 = conv3x3x3(ic, oc, stride)
        self.bn1 = norm_layer(oc)
        self.act = act_layer(inplace=True)
        self.conv2 = conv3x3x3(oc, oc)
        self.bn2 = norm_layer(oc)

        self.drop_path= DropPath(drop_prob=drop_path_rate)

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv3d(
                    ic * expansion_factor, 
                    oc,
                    kernel_size=(1, 1, 1), 
                    stride=(2,2,2), 
                    bias=False
                    ),
                norm_layer(oc),
            )
        else:
            self.downsample= nn.Identity()

    def forward(self, x):        
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = self.drop_path(x)

        residual = self.downsample(residual)
        x += residual
        x = self.act(x)

        return x

class Bottleneck(nn.Module):
    def __init__(
        self, 
        ic, 
        oc, 
        stride: int = 1, 
        downsample: bool = None, 
        expansion_factor: int = 4,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module = nn.BatchNorm3d,
        act_layer: nn.Module = nn.ReLU,
    ):
        super().__init__()

        self.conv1 = nn.Conv3d(ic * expansion_factor, oc, kernel_size=1, bias=False)
        self.bn1 = norm_layer(oc)
        self.conv2 = nn.Conv3d(oc, oc, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(oc)
        self.conv3 = nn.Conv3d(oc, oc * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(oc * 4)
        self.act = act_layer(inplace=True)

        self.drop_path= DropPath(drop_prob=drop_path_rate)

        if downsample is not None:
            stride = (1,1,1) if expansion_factor == 1 else (2,2,2)
            self.downsample = nn.Sequential(
                nn.Conv3d(ic * expansion_factor, oc * 4, kernel_size=(1, 1, 1), stride=stride, bias=False),
                norm_layer(oc * 4),
            )
        else:
            self.downsample= nn.Identity()

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = self.drop_path(x)

        residual = self.downsample(residual)
        x += residual
        x = self.act(x)

        return x


class ResnetEncoder3d(nn.Module):
    def __init__(
        self, 
        cfg: SimpleNamespace,
        inference_mode: bool = False,
        drop_path_rate: float = 0.2,
        in_stride: tuple[int]= (2,2,2),
        in_dilation: tuple[int]= (1,1,1),
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.cfg= cfg
        self.use_checkpoint= use_checkpoint

        # Backbone configs
        bb= self.cfg.backbone
        backbone_cfg= {
            "r3d18": ([2, 2, 2, 2], BasicBlock),
            "r3d200": ([3, 24, 36, 3], Bottleneck),
        }
        if bb in backbone_cfg:
            layers, block = backbone_cfg[bb]
            wpath = "/content/data/model_zoo/{}_KM_200ep.pt".format(bb)
        else:
            raise ValueError(f"ResnetEncoder3d backbone: {bb} not implemented.")

        # Drop_path_rates (linearly scaled)
        num_blocks = sum(layers)
        flat_drop_path_rates = [drop_path_rate * (i / (num_blocks - 1)) for i in range(num_blocks)]
        drop_path_rates = []
        start = 0
        for b in layers:
            end = start + b
            drop_path_rates.append(flat_drop_path_rates[start:end])
            start = end

        # Stem
        in_padding= tuple(_*3 for _ in in_dilation)
        self.conv1 = nn.Conv3d(
            in_channels= 3, 
            out_channels= 64,
            kernel_size= (7, 7, 7), 
            stride= in_stride, 
            dilation= in_dilation,
            padding= in_padding, 
            bias= False,
            )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        # Blocks
        self.layer1 = self._make_layer(
            ic=64, oc=64, block=block, n_blocks=layers[0], stride=1, downsample=False, 
            drop_path_rates= drop_path_rates[0],
            )

        self.layer2 = self._make_layer(
            ic=64, oc=128, block=block, n_blocks=layers[1], stride=2, downsample=True,
            drop_path_rates=drop_path_rates[1],
            )

        self.layer3 = self._make_layer(
            ic=128, oc=256, block=block, n_blocks=layers[2], stride=2, downsample=True,
            drop_path_rates=drop_path_rates[2],
            )

        self.layer4 = self._make_layer(
            ic=256, oc=512, block=block, n_blocks=layers[3], stride=2, downsample=True,
            drop_path_rates=drop_path_rates[3],
            )

        # Load pretrained weights
        if not inference_mode:
            load_weights(self, wpath)

        # In channels
        self._update_input_channels()

        # Encoder channels
        with torch.no_grad():
            out = self.forward_features(torch.randn((1, self.cfg.in_chans, 96, 96, 96)))
            self.channels = [o.shape[1] for o in out]
            del out

    def _make_layer(
        self, ic, oc, block, n_blocks, stride=1, downsample=False, 
        drop_path_rates=[],
        ):
        layers = []
        if downsample:
            layers.append(
                block(
                    ic=ic, oc=oc, stride=stride, downsample=downsample, 
                    drop_path_rate=drop_path_rates[0],
                    ),
                )
        else:
            layers.append(
                block(
                    ic=ic, oc=oc, stride=stride, downsample=downsample, expansion_factor=1,
                    drop_path_rate=drop_path_rates[0],
                    ),
                )
        
        for i in range(1, n_blocks):
            layers.append(block(oc, oc, drop_path_rate=drop_path_rates[i]))

        return nn.Sequential(*layers)

    def _update_input_channels(self, ):
        with torch.no_grad():
            # Get stem
            b= self.conv1

            # Update channels
            ic= self.cfg.in_chans
            b.in_channels = ic
            w = b.weight.sum(dim=1, keepdim=True) / ic
            b.weight = nn.Parameter(w.repeat([1, ic] + [1] * (w.ndim - 2)))
        return

    def _checkpoint_if_enabled(self, module, x):
        return checkpoint(module, x) if self.use_checkpoint else module(x)

    def forward_features(self, x):
        res= []

        # Stem
        x = self._checkpoint_if_enabled(self.conv1, x)
        x = self.bn1(x)
        x = self.relu(x)
        res.append(x)
        x = self.maxpool(x)

        # Layers
        layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        for layer in layers:
            x = self._checkpoint_if_enabled(layer, x)
            res.append(x)

        return res

    def forward(self, x):
        # Stem
        x = self._checkpoint_if_enabled(self.conv1, x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Layers
        layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        for layer in layers:
            x = self._checkpoint_if_enabled(layer, x)
        return x


if __name__ == "__main__":
    from .utils import count_parameters

    cfg= SimpleNamespace()
    cfg.backbone= "r3d18"
    # cfg.backbone= "r3d200"
    cfg.in_chans= 1
    cfg.encoder_cfg= SimpleNamespace()
    cfg.encoder_cfg.use_checkpoint= True
    cfg.roi_size= (32, 128, 128)

    m = ResnetEncoder3d(
        cfg= cfg,
        inference_mode= False,
        **vars(cfg.encoder_cfg),
    ).eval()

    # Param count
    n_params= count_parameters(m)
    print(f"Model: {type(m).__name__}")
    print("n_param: {:_}".format(n_params))

    # Normal
    x = torch.ones(8, cfg.in_chans, *cfg.roi_size)
    with torch.no_grad():
        z = m.forward_features(x)

    print(x.shape)
    print([_.shape for _ in z])