"""UNet3D + Temporal Attention + SimpleNodeTransformer for cell tracking.

Architecture:
  - ResnetEncoder3d backbone (R3D-200, trained from scratch or pretrained)
  - _TemporalAttention at the bottleneck (deepest encoder stage)
  - UnetDecoder3d with skip connections
  - SegmentationHead3d → per-frame detection heatmap logits
  - SimpleNodeTransformer → edge logits between consecutive frames
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._base import BaseModel
from src.models.layers.resnet3d import ResnetEncoder3d
from src.models.layers.unet3d import UnetDecoder3d, SegmentationHead3d
from src.models.layers.transformer import SimpleNodeTransformer


class _TemporalAttention(nn.Module):
    """Per-voxel multi-head self-attention across W frames at the bottleneck."""

    def __init__(self, channels: int, n_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, *spatial)
        B, T, C = x.shape[:3]
        spatial = x.shape[3:]
        S = math.prod(spatial)

        h = x.reshape(B, T, C, S).permute(0, 3, 1, 2).reshape(B * S, T, C)
        h = self.norm(h)
        h, _ = self.attn(h, h, h, need_weights=False)
        h = h.reshape(B, S, T, C).permute(0, 2, 3, 1).reshape(B, T, C, *spatial)
        return x + h


class Net(BaseModel):
    """Cell tracking model: ResNet encoder + temporal attention + UNet decoder + edge transformer."""

    def __init__(self, cfg, inference_mode: bool = False):
        super().__init__(cfg, inference_mode)

        # Backbone: always skip pre-trained CT weights for cell microscopy
        self.backbone = ResnetEncoder3d(cfg, inference_mode=True)

        # Temporal attention at bottleneck (deepest encoder stage)
        bottleneck_ch = self.backbone.channels[-1]
        n_temporal_heads = getattr(cfg, "n_temporal_heads", 4)
        self.temporal_attn = _TemporalAttention(bottleneck_ch, n_heads=n_temporal_heads)

        # UNet decoder: encoder_channels reversed (deep → shallow)
        enc_ch_reversed = self.backbone.channels[::-1]  # e.g. [2048, 1024, 512, 256, 64]
        decoder_channels = tuple(getattr(cfg, "decoder_channels", (256, 128, 64, 32)))
        scale_factors = tuple(getattr(cfg, "scale_factors", (2, 2, 2, 2)))
        self.decoder = UnetDecoder3d(
            encoder_channels=enc_ch_reversed,
            decoder_channels=decoder_channels,
            scale_factors=scale_factors,
            upsample_mode="nontrainable",
        )

        # Detection head: conv + 2x upsample → full downsampled resolution
        det_in_ch = decoder_channels[-1]
        self.det_head = SegmentationHead3d(
            in_channels=det_in_ch,
            out_channels=1,
            scale_factor=(2, 2, 2),
        )

        # Feature dim for transformer = det_in_ch (indexed features) + pos_embed_dim
        pos_embed_dim = getattr(cfg, "pos_embed_dim", 8) * 4  # 4 axes
        feat_dim = det_in_ch + pos_embed_dim

        self.transformer = SimpleNodeTransformer(
            feat_dim=feat_dim,
            hidden_dim=getattr(cfg, "transformer_hidden_dim", 128),
            n_heads=getattr(cfg, "n_heads", 4),
            n_blocks=getattr(cfg, "n_blocks", 4),
            dropout=getattr(cfg, "dropout", 0.3),
            pair_chunk_size=getattr(cfg, "pair_chunk_size", 32),
        )

        # Scale for indexing dec_feats[-1] (at 1/2 of full downsampled resolution)
        self._feat_scale = 0.5

    # ------------------------------------------------------------------
    def _index_features(
        self,
        feat_map: torch.Tensor,   # (B, C, Z, Y, X)
        coords: torch.Tensor,     # (B, M, 3) in downsampled space
        mask: torch.Tensor,       # (B, M) bool
    ) -> torch.Tensor:
        """Integer-index feat_map at scaled GT positions; padded slots → zeros.

        Coordinates are halved because dec_feats[-1] is at 1/2 the input resolution.
        Gradients flow through values, not through the integer coordinate indices.
        """
        B, C = feat_map.shape[:2]
        spatial = feat_map.shape[2:]
        M = coords.shape[1]
        out = torch.zeros(B, M, C, device=feat_map.device, dtype=feat_map.dtype)

        for b in range(B):
            n = int(mask[b].sum().item())
            if n == 0:
                continue
            z = (coords[b, :n, 0] * self._feat_scale).long().clamp(0, spatial[0] - 1)
            y = (coords[b, :n, 1] * self._feat_scale).long().clamp(0, spatial[1] - 1)
            x = (coords[b, :n, 2] * self._feat_scale).long().clamp(0, spatial[2] - 1)
            out[b, :n] = feat_map[b, :, z, y, x].T  # (n, C)

        return out  # (B, M, C)

    # ------------------------------------------------------------------
    def encode(self, imgs: torch.Tensor):
        """Encode W frames through backbone + temporal attention + decoder.

        Parameters
        ----------
        imgs : (B, W, 1, Z_ds, Y_ds, X_ds)

        Returns
        -------
        det_feats : (B*W, C_dec, Z_ds/2, Y_ds/2, X_ds/2)   decoder last block
        det_logits : (B*W, 1, Z_ds, Y_ds, X_ds)             detection logits
        B, W : batch and window sizes
        """
        B, W = imgs.shape[:2]
        x = imgs.reshape(B * W, *imgs.shape[2:])  # (B*W, 1, Z, Y, X)

        feats = self.backbone.forward_features(x)  # [stem, l1, l2, l3, l4]

        # Temporal attention at bottleneck (feats[-1] = layer4)
        l4 = feats[-1]  # (B*W, C4, z4, y4, x4)
        l4 = l4.reshape(B, W, *l4.shape[1:])      # (B, W, C4, ...)
        l4 = self.temporal_attn(l4)               # (B, W, C4, ...)
        l4 = l4.reshape(B * W, *l4.shape[2:])    # (B*W, C4, ...)
        feats[-1] = l4

        # Decoder: expects [deepest, ..., shallowest]
        enc_feats_rev = feats[::-1]               # [l4, l3, l2, l1, stem]
        dec_feats = self.decoder(enc_feats_rev)   # list of decoder block outputs

        det_feats = dec_feats[-1]                 # (B*W, C_dec, Z/2, Y/2, X/2)
        det_logits = self.det_head(det_feats)     # (B*W, 1, Z, Y, X)

        return det_feats, det_logits, B, W

    # ------------------------------------------------------------------
    def forward(self, batch: dict) -> dict:
        imgs = batch["input"]          # (B, W, 1, Z, Y, X)
        target = batch["target"]       # (B, W, 1, Z, Y, X)
        coords = batch["coords"]       # (B, W, M, 3)
        masks = batch["masks"]         # (B, W, M) bool
        edge_targets = batch["edge_targets"]  # (B, W-1, M, M)
        pos_feats = batch["pos_feats"]         # (B, W, M, D)

        # ----------------------------------------------------------------
        # 1. Encode all W frames
        # ----------------------------------------------------------------
        det_feats_bw, det_logits_bw, B, W = self.encode(imgs)

        det_logits = det_logits_bw.reshape(B, W, *det_logits_bw.shape[1:])
        det_feats = det_feats_bw.reshape(B, W, *det_feats_bw.shape[1:])

        # ----------------------------------------------------------------
        # 2. Index decoder features at GT positions, cat pos-embeddings
        # ----------------------------------------------------------------
        node_feats = []
        for t in range(W):
            f = self._index_features(det_feats[:, t], coords[:, t], masks[:, t])  # (B, M, C)
            node_feats.append(torch.cat([f, pos_feats[:, t]], dim=-1))             # (B, M, C+D)

        # ----------------------------------------------------------------
        # 3. Predict edges for consecutive frame pairs
        # ----------------------------------------------------------------
        logits_edge = []
        for t in range(W - 1):
            e = self.transformer(
                node_feats[t], node_feats[t + 1],
                coords[:, t], coords[:, t + 1],
                masks[:, t], masks[:, t + 1],
            )
            logits_edge.append(e)  # (B, M, M)

        # ----------------------------------------------------------------
        # 4. Loss
        # ----------------------------------------------------------------
        total_loss, det_loss, edge_loss = self.loss_fn(
            det_logits, target, logits_edge, edge_targets, masks,
        )

        return {
            "loss": total_loss,
            "det_loss": det_loss,
            "edge_loss": edge_loss,
            "logits_edge": [e.detach() for e in logits_edge],  # for eval metrics
            "masks": masks,
            "edge_targets": edge_targets,
        }
