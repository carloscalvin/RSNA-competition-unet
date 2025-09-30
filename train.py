import argparse
from copy import copy
import sys
import importlib
import os
import random
import numpy as np

import torch
import torch.distributed as dist

from src.utils.cfg import update_cfg
from src.modules.train import train

def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def parse_args(local_rank: int = 0):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-C", "--config", help="config filename", default="r3d200")
    parser.add_argument("-G", "--gpu_id", default="", help="GPU ID")
    parser_args, other_args = parser.parse_known_args(sys.argv)

    # Use all GPUs unless specified
    if parser_args.gpu_id != "":
        os.environ['CUDA_VISIBLE_DEVICES'] = str(parser_args.gpu_id)

    # Load CFG
    cfg = copy(importlib.import_module('src.configs.{}'.format(parser_args.config)).cfg)
    cfg.config_file = parser_args.config
    if local_rank == 0:
        print("config ->", cfg.config_file)

    # Update args
    if len(other_args) > 1:
        other_args = {v.split("=")[0].lstrip("-"):v.split("=")[1] for v in other_args[1:]}
        cfg= update_cfg(
            cfg=cfg, other_args=other_args, log=local_rank == 0,
            )

    # Set seed
    if cfg.seed < 0:
        cfg.seed = np.random.randint(1_000_000)
    if local_rank == 0:
        print("seed", cfg.seed)
    set_seed(cfg.seed)

    # Quick development run
    if cfg.fast_dev_run:
        cfg.epochs= 1
        cfg.no_wandb= None

    return cfg

def setup(rank, world_size):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    return

def cleanup():
    dist.barrier()
    dist.destroy_process_group()
    return

def is_torchrun():
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ

if __name__ == "__main__":

    # Multi-GPU
    if is_torchrun():
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(f"Rank: {rank}, World size: {world_size}")

        # Args
        cfg = parse_args(local_rank=rank)
        cfg.local_rank= rank
        cfg.world_size= world_size

        # Not working w/ DDP
        for attr in ['ema', 'use_checkpoint']:
            if hasattr(cfg, attr):
                setattr(cfg, attr, False)
        if hasattr(cfg, 'encoder_cfg') and hasattr(cfg.encoder_cfg, 'use_checkpoint'):
            cfg.encoder_cfg.use_checkpoint = False

        setup(rank, world_size)
        train(cfg)
        cleanup()

    # Single-GPU
    else:
        cfg = parse_args()
        cfg.local_rank = 0
        cfg.world_size = 1

        train(cfg)
