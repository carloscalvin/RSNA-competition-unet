"""Combined detection + edge loss for cell tracking."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import SmoothBCE


def _edge_loss_pair(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Focal BCE on annotated rows/cols; softmax over dim=0 (allows divisions)."""
    active_rows = target.sum(dim=1) > 0
    active_cols = target.sum(dim=0) > 0
    mask = active_rows.unsqueeze(1) | active_cols.unsqueeze(0)
    if not mask.any():
        return torch.zeros(1, device=logits.device, requires_grad=True).squeeze()

    probs = torch.softmax(logits, dim=0)
    bce = F.binary_cross_entropy(probs, target, reduction="none")
    p_t = probs * target + (1 - probs) * (1 - target)
    return (((1 - p_t) ** 2) * bce)[mask].mean()


def _edge_loss_batch(
    logits: torch.Tensor,   # (B, M, M)
    target: torch.Tensor,   # (B, M, M)
    mask_t: torch.Tensor,   # (B, M) bool
    mask_t1: torch.Tensor,  # (B, M) bool
) -> torch.Tensor:
    losses = []
    for b in range(logits.shape[0]):
        nt = int(mask_t[b].sum().item())
        nt1 = int(mask_t1[b].sum().item())
        if nt == 0 or nt1 == 0:
            losses.append(torch.zeros(1, device=logits.device).squeeze())
        else:
            losses.append(_edge_loss_pair(logits[b, :nt, :nt1], target[b, :nt, :nt1]))
    return torch.stack(losses).mean()


class TrackingLoss(nn.Module):
    """Heatmap detection (SmoothBCE) + edge prediction (focal-softmax BCE)."""

    def __init__(
        self,
        det_weight: float = 1.0,
        edge_weight: float = 1.0,
        det_pos_weight: float = 256.0,
        det_smooth: float = 1e-3,
    ):
        super().__init__()
        self.det_weight = det_weight
        self.edge_weight = edge_weight
        self.det_loss_fn = SmoothBCE(smooth=det_smooth, pos_weight=det_pos_weight)

    def forward(
        self,
        logits_det: torch.Tensor,           # (B, W, 1, Z, Y, X)
        targets_heatmap: torch.Tensor,       # (B, W, 1, Z, Y, X)
        logits_edge: list[torch.Tensor],     # W-1 × (B, M, M)
        edge_targets: torch.Tensor,          # (B, W-1, M, M)
        masks: torch.Tensor,                 # (B, W, M) bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, W = logits_det.shape[:2]

        det_loss = self.det_loss_fn(
            logits_det.reshape(B * W, *logits_det.shape[2:]),
            targets_heatmap.reshape(B * W, *targets_heatmap.shape[2:]),
        )

        if logits_edge:
            edge_loss = torch.stack([
                _edge_loss_batch(
                    logits_edge[i], edge_targets[:, i],
                    masks[:, i], masks[:, i + 1],
                )
                for i in range(W - 1)
            ]).mean()
        else:
            edge_loss = torch.zeros(1, device=logits_det.device).squeeze()

        total = self.det_weight * det_loss + self.edge_weight * edge_loss
        return total, det_loss.detach(), edge_loss.detach()
