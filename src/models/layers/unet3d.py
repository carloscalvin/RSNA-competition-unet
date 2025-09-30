import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks import UpSample

class ConvBnAct3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding: int = 0,
        stride: int = 1,
        norm_layer: nn.Module = nn.BatchNorm3d,
        act_layer: nn.Module = nn.ReLU,
    ):
        super().__init__()

        self.conv= nn.Conv3d(
            in_channels, 
            out_channels,
            kernel_size,
            stride=stride, 
            padding=padding, 
            bias=False,
        )
        self.norm = norm_layer(out_channels)
        self.act= act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class DecoderBlock3d(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer: nn.Module = nn.BatchNorm3d,
        upsample_mode: str = "deconv",
        scale_factor: int = 2,
    ):
        super().__init__()
        
        self.upsample = UpSample(
            spatial_dims= 3,
            in_channels= in_channels,
            out_channels= in_channels,
            scale_factor= scale_factor,
            mode= upsample_mode,
        )

        self.conv1 = ConvBnAct3d(
            in_channels + skip_channels,
            out_channels,
            kernel_size= 3,
            padding= 1,
            norm_layer= norm_layer,
        )

        self.conv2 = ConvBnAct3d(
            out_channels,
            out_channels,
            kernel_size= 3,
            padding= 1,
            norm_layer= norm_layer,
        )


    def forward(self, x, skip: torch.Tensor = None):
        x = self.upsample(x)

        if skip is not None:
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UnetDecoder3d(nn.Module):
    def __init__(
        self,
        encoder_channels: tuple[int],
        skip_channels: tuple[int] = None,
        decoder_channels: tuple[int] = (256, 128, 64, 32, 16),
        scale_factors: tuple[int]= (2,2,2,2,2),
        norm_layer: nn.Module = nn.BatchNorm3d,
        attention_type: str = None,
        intermediate_conv: bool = False,
        upsample_mode: str = "nontrainable",
    ):
        super().__init__()
        self.decoder_channels= decoder_channels
        
        if skip_channels is None:
            skip_channels= list(encoder_channels[1:]) + [0]

        # Build decoder blocks
        in_channels= [encoder_channels[0]] + list(decoder_channels[:-1])
        self.blocks = nn.ModuleList()

        for i, (ic, sc, dc, sf) in enumerate(zip(
            in_channels, skip_channels, decoder_channels, scale_factors,
            )):
            self.blocks.append(
                DecoderBlock3d(
                    ic, sc, dc, 
                    norm_layer= norm_layer,
                    upsample_mode= upsample_mode,
                    scale_factor= sf,
                    )
            )

    def forward(self, feats: list[torch.Tensor]):
        res= [feats[0]]
        feats= feats[1:]

        for i, b in enumerate(self.blocks):
            skip= feats[i] if i < len(feats) else None
            res.append(
                b(res[-1], skip=skip),
                )
            
        return res

class SegmentationHead3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        scale_factor: tuple[int] = (2,2,2),
    ):
        super().__init__()
        self.conv= nn.Conv3d(
            in_channels, out_channels, 
            kernel_size = 3, padding = 1,
        )

        self.upsample = UpSample(
            spatial_dims= 3,
            in_channels= out_channels,
            out_channels= out_channels,
            scale_factor= scale_factor,
            mode= "nontrainable",
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        return x


if __name__ == "__main__":

    m= UnetDecoder3d(
        encoder_channels=[128, 64, 32, 16, 8],
    )
    m.cuda().eval()

    with torch.no_grad():
        x= [
            torch.ones([2, 128, 1, 4, 4]).cuda(), 
            torch.ones([2, 64, 2, 8, 8]).cuda(), 
            torch.ones([2, 32, 4, 16, 16]).cuda(), 
            torch.ones([2, 16, 8, 32, 32]).cuda(), 
            torch.ones([2, 8, 16, 64, 64]).cuda(),
            ]

        z = m(x)
        print([_.shape for _ in z])