"""Cell-tracking Dataset: zarr/geff → padded tensors for RSNA training loop.

df must have column "ds_path" pointing to the base path of each dataset
(without extension; the loader appends .zarr / .geff automatically).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset
import polars as pl
from tracking_cellmot.io import open_dataset

from src.data._3d import create_heatmap_from_coords

_POS_EMBED_DIM = 8  # per axis → total 4 * _POS_EMBED_DIM = 32 dims


@dataclass
class _VideoMeta:
    zarr_path: Path
    image_shape: tuple[int, ...]  # (T, Z_ds, Y_ds, X_ds)
    downsample: tuple[int, ...]
    q_low: float
    q_high: float


@dataclass
class _WindowMeta:
    video_idx: int
    t_start: int
    pos_feats: list[torch.Tensor]     # W × (M_i, D)
    coords: list[torch.Tensor]        # W × (M_i, 3) in downsampled space
    node_counts: list[int]
    edge_targets: list[torch.Tensor]  # (W-1) × (M_t, M_t1) transition matrices


def _sinusoidal_pos_embed(coords: np.ndarray, image_shape: tuple,
                           dim: int = _POS_EMBED_DIM) -> np.ndarray:
    """Sinusoidal pos-embed for (t, z, y, x) normalised by image_shape."""
    norms = [coords[:, i] / max(image_shape[i], 1) for i in range(4)]

    def _embed(vals: np.ndarray) -> np.ndarray:
        freqs = 2 ** np.arange(dim // 2)
        angles = vals[:, None] * freqs * np.pi
        return np.concatenate([np.sin(angles), np.cos(angles)], axis=1)

    return np.concatenate([_embed(n) for n in norms], axis=1).astype(np.float32)


def _transition_matrix(gt_ids_t: np.ndarray, gt_ids_t1: np.ndarray,
                        edge_attrs: pl.DataFrame) -> torch.Tensor:
    t_to_row = {nid: i for i, nid in enumerate(gt_ids_t)}
    t1_to_col = {nid: i for i, nid in enumerate(gt_ids_t1)}
    mat = torch.zeros(len(gt_ids_t), len(gt_ids_t1), dtype=torch.float32)
    for src, tgt in zip(edge_attrs["source_id"], edge_attrs["target_id"]):
        if src in t_to_row and tgt in t1_to_col:
            mat[t_to_row[src], t1_to_col[tgt]] = 1.0
    return mat


class CustomDataset(Dataset):
    """Cell tracking dataset compatible with the RSNA training loop.

    Returns a dict with keys:
        input       (W, 1, Z, Y, X)      – normalised frames
        target      (W, 1, Z, Y, X)      – Gaussian heatmaps
        coords      (W, M, 3)            – GT coords in downsampled space
        masks       (W, M) bool          – which slots are real nodes
        edge_targets (W-1, M, M) float   – GT transition matrices
        pos_feats   (W, M, D) float      – sinusoidal pos-embeddings
        downsample  (3,) float
    """

    def __init__(self, cfg, df, mode: str = "train"):
        self.cfg = cfg
        self.mode = mode
        self.window_size = int(cfg.window_size)
        self.downsample = tuple(int(d) for d in cfg.downsample)
        self._videos: list[_VideoMeta] = []
        self._windows: list[_WindowMeta] = []

        for ds_path in df["ds_path"].tolist():
            self._load_video(Path(ds_path))

        if not self._windows:
            raise RuntimeError(
                f"No valid windows found across {len(df)} dataset(s). "
                "Check that .zarr + .geff files exist and have GT annotations."
            )

        self.max_nodes: int = getattr(cfg, "max_nodes", None) or max(
            max(w.node_counts) for w in self._windows
        )

    # ------------------------------------------------------------------
    def _load_video(self, ds_path: Path) -> None:
        dz, dy, dx = self.downsample
        try:
            ds = open_dataset(
                ds_path, normalize=False, require_tracks=True,
                load_image=False, downsample=(dz, dy, dx),
            )
        except FileNotFoundError as e:
            print(f"Warning: skipping {ds_path}: {e}")
            return

        if "0.001" not in ds.quantiles or "0.999" not in ds.quantiles:
            print(f"Warning: {ds_path} missing image_statistics.quantiles — skipping.")
            return

        v_idx = len(self._videos)
        self._videos.append(_VideoMeta(
            zarr_path=ds.zarr_path,
            image_shape=ds.image_shape,
            downsample=self.downsample,
            q_low=float(ds.quantiles["0.001"]),
            q_high=float(ds.quantiles["0.999"]),
        ))

        self._build_windows(v_idx, ds.tracks, ds.image_shape)

    def _build_windows(self, v_idx: int, tracks, image_shape: tuple) -> None:
        dz, dy, dx = self.downsample
        ds_arr = np.array([dz, dy, dx], dtype=np.float32)
        W = self.window_size
        T = image_shape[0]

        gt_attrs = tracks.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
        edge_attrs = tracks.edge_attrs(attr_keys=["source_id", "target_id"])

        for t_start in range(T - W + 1):
            meta = self._build_one_window(
                gt_attrs, edge_attrs, image_shape, t_start, W, ds_arr, v_idx
            )
            if meta is not None:
                self._windows.append(meta)

    def _build_one_window(self, gt_attrs, edge_attrs, image_shape,
                           t_start, W, ds_arr, v_idx) -> _WindowMeta | None:
        per_frame_ids: list[np.ndarray] = []
        pos_feats_list: list[torch.Tensor] = []
        coords_list: list[torch.Tensor] = []
        node_counts: list[int] = []

        for i in range(W):
            t = t_start + i
            gt_t = gt_attrs.filter(pl.col("t") == t)
            if len(gt_t) == 0:
                return None  # skip windows with unannotated frames

            gt_coords = gt_t.select(["z", "y", "x"]).to_numpy().astype(np.float32) / ds_arr
            gt_ids = gt_t["node_id"].to_numpy()
            n = len(gt_coords)

            t_col = np.full((n, 1), float(i), dtype=np.float32)
            full_coords = np.concatenate([t_col, gt_coords], axis=1)
            pos = _sinusoidal_pos_embed(full_coords, image_shape)

            pos_feats_list.append(torch.from_numpy(pos))
            coords_list.append(torch.from_numpy(gt_coords))
            per_frame_ids.append(gt_ids)
            node_counts.append(n)

        edge_targets_list = [
            _transition_matrix(per_frame_ids[i], per_frame_ids[i + 1], edge_attrs)
            for i in range(W - 1)
        ]

        return _WindowMeta(
            video_idx=v_idx,
            t_start=t_start,
            pos_feats=pos_feats_list,
            coords=coords_list,
            node_counts=node_counts,
            edge_targets=edge_targets_list,
        )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        meta = self._windows[idx]
        vm = self._videos[meta.video_idx]
        W = self.window_size
        M = self.max_nodes
        D = meta.pos_feats[0].shape[1]

        # --- Load W frames from zarr ---
        dz, dy, dx = vm.downsample
        z_store = zarr.open_group(str(vm.zarr_path), mode="r")["0"]
        raw = z_store[meta.t_start:meta.t_start + W, ::dz, ::dy, ::dx].astype(np.float32)
        imgs = torch.from_numpy(
            (raw - vm.q_low) / (vm.q_high - vm.q_low + 1e-6)
        ).clamp(0.0)  # (W, Z_ds, Y_ds, X_ds)

        # Resize to expected spatial shape if strided I/O is slightly off
        target_spatial = list(vm.image_shape[1:])
        if list(imgs.shape[1:]) != target_spatial:
            import torch.nn.functional as Fn
            imgs = Fn.interpolate(
                imgs[:, None].float(), size=target_spatial,
                mode="trilinear", align_corners=False,
            )[:, 0]

        imgs = imgs.unsqueeze(1).float()  # (W, 1, Z_ds, Y_ds, X_ds)

        # --- Augmentation (training only) ---
        coords_aug = [c.clone() for c in meta.coords]
        masks_list = [torch.ones(n, dtype=torch.bool) for n in meta.node_counts]

        if self.mode == "train":
            imgs, coords_aug, masks_list = self._augment(imgs, coords_aug, masks_list)

        # --- Gaussian heatmaps ---
        Z_ds, Y_ds, X_ds = imgs.shape[2], imgs.shape[3], imgs.shape[4]
        heatmaps = []
        for t in range(W):
            n = meta.node_counts[t]
            c_t = coords_aug[t]
            coords_int = [
                (0,
                 int(round(float(c_t[i, 0]))),
                 int(round(float(c_t[i, 1]))),
                 int(round(float(c_t[i, 2]))))
                for i in range(n)
            ]
            hm = create_heatmap_from_coords(
                coords_int,
                target_shape=(1, Z_ds, Y_ds, X_ds),
                kernel_size=self.cfg.kernel_size,
                kernel_sigma=self.cfg.kernel_sigma,
            )
            heatmaps.append(torch.from_numpy(hm))
        target = torch.stack(heatmaps, dim=0)  # (W, 1, Z_ds, Y_ds, X_ds)

        # --- Pad to (W, M, ...) ---
        pos_feats_pad = torch.zeros(W, M, D, dtype=torch.float32)
        coords_pad = torch.zeros(W, M, 3, dtype=torch.float32)
        masks_pad = torch.zeros(W, M, dtype=torch.bool)

        for t in range(W):
            n = meta.node_counts[t]
            pos_feats_pad[t, :n] = meta.pos_feats[t]
            coords_pad[t, :n] = coords_aug[t]
            masks_pad[t, :n] = masks_list[t][:n]

        edge_pad = torch.zeros(W - 1, M, M, dtype=torch.float32)
        for t in range(W - 1):
            nt = meta.node_counts[t]
            nt1 = meta.node_counts[t + 1]
            edge_pad[t, :nt, :nt1] = meta.edge_targets[t]

        return {
            "input": imgs,               # (W, 1, Z, Y, X)
            "target": target,            # (W, 1, Z, Y, X)
            "coords": coords_pad,        # (W, M, 3)
            "masks": masks_pad,          # (W, M)
            "edge_targets": edge_pad,    # (W-1, M, M)
            "pos_feats": pos_feats_pad,  # (W, M, D)
            "downsample": torch.tensor(self.downsample, dtype=torch.float32),
        }

    def _augment(self, imgs, coords, masks):
        from src.augs.aug3d import temporal_flip, temporal_brightness, temporal_coarse_dropout
        rng = np.random.default_rng()
        imgs, coords, masks = temporal_flip(imgs, coords, masks, rng)
        imgs, coords, masks = temporal_brightness(imgs, coords, masks, rng)
        imgs, coords, masks = temporal_coarse_dropout(imgs, coords, masks, rng)
        return imgs, coords, masks
