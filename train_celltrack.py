"""Entry point for cell-tracking training.

Usage:
    python train_celltrack.py --config src.configs.celltrack

The config must define:
    cfg.splits_json  — path to JSON with {"train": [...], "val": [...]} dataset paths
    cfg.model_type   — "unet3d_celltrack"
    cfg.dataset_type — "_3d_celltrack"
"""
import os
import sys
import argparse
import random
from importlib import import_module

import numpy as np
import torch

from src.modules.train import train_celltrack


def parse_args():
    parser = argparse.ArgumentParser(description="Cell Tracking Training")
    parser.add_argument("--config", type=str, default="src.configs.celltrack",
                        help="Python import path to the config module")
    parser.add_argument("--fast_dev_run", action="store_true",
                        help="Run one mini-batch to verify the pipeline")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (overrides config)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    cfg = import_module(args.config).cfg

    # Overrides from CLI
    if args.fast_dev_run:
        cfg.fast_dev_run = True
    if args.seed is not None:
        cfg.seed = args.seed

    # Seed everything
    seed = cfg.seed if cfg.seed >= 0 else 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Single-GPU / torchrun multi-GPU
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        cfg.local_rank = int(os.environ["LOCAL_RANK"])
        cfg.world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(cfg.local_rank)
        torch.distributed.init_process_group("nccl")
    else:
        cfg.local_rank = 0
        cfg.world_size = 1

    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.local_rank == 0:
        print(f"Config: {args.config}")
        print(f"Device: {cfg.device}  |  World size: {cfg.world_size}")
        print(f"Fast dev run: {cfg.fast_dev_run}")

    train_celltrack(cfg)


if __name__ == "__main__":
    main()
