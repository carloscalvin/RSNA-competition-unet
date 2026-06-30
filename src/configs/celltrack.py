"""Cell-tracking config — extends _base.py defaults."""
from types import SimpleNamespace

from src.configs._base import cfg

# -------------------------------------------------------------------
# Project / paths
# -------------------------------------------------------------------
cfg.project = "biohub_cell_tracking"

# Path to JSON file with train/val dataset splits.
# Format: {"train": ["/path/to/ds1", ...], "val": ["/path/to/ds2", ...]}
cfg.splits_json = "data/splits.json"

# Directory to save model checkpoints
cfg.save_dir = "checkpoints/celltrack"

# -------------------------------------------------------------------
# Model
# -------------------------------------------------------------------
cfg.model_type = "unet3d_celltrack"
cfg.backbone = "r3d200"     # ResNet3D-200
cfg.in_chans = 1            # grayscale microscopy

# Decoder channels (4 blocks: l4→l3→l2→l1→stem then SegHead×2)
cfg.decoder_channels = (256, 128, 64, 32)
cfg.scale_factors = (2, 2, 2, 2)

# Temporal attention at bottleneck
cfg.n_temporal_heads = 4

# SimpleNodeTransformer
cfg.transformer_hidden_dim = 128
cfg.n_heads = 4
cfg.n_blocks = 4
cfg.dropout = 0.3
cfg.pair_chunk_size = 32     # chunk size for pairwise MLP (saves memory)
cfg.pos_embed_dim = 8        # sinusoidal dims per axis; total = 4 * 8 = 32

# -------------------------------------------------------------------
# Loss
# -------------------------------------------------------------------
cfg.loss_type = "src.losses.TrackingLoss"
cfg.loss_cfg = SimpleNamespace(
    det_weight=1.0,
    edge_weight=1.0,
    det_pos_weight=256.0,
    det_smooth=1e-3,
)

# -------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------
cfg.dataset_type = "_3d_celltrack"
cfg.window_size = 2           # W consecutive frames per sample
cfg.downsample = (1, 4, 4)    # (dz, dy, dx) spatial downsample strides

# Heatmap generation (detection target)
cfg.kernel_size = 9
cfg.kernel_sigma = 1.5

# Detection / NMS (inference)
cfg.det_threshold = 0.3
cfg.nms_radius = 3

# max_nodes: padded GT slots per frame. None → auto (max across training set)
cfg.max_nodes = None

# -------------------------------------------------------------------
# Training
# -------------------------------------------------------------------
cfg.epochs = 100
cfg.batch_size = 4           # W frames × batch → fits A100 80 GB
cfg.batch_size_val = 2
cfg.num_workers = 8
cfg.lr = 1e-4
cfg.lr_min = 1e-6
cfg.weight_decay = 1e-4
cfg.scheduler = "CosineAnnealingLR"
cfg.ema = True
cfg.ema_decay = 0.995
cfg.mixed_precision = True    # bfloat16 on A100
cfg.grad_clip = 1.0
cfg.eval_epochs = 5           # validate every N epochs
cfg.logging_steps = 10
cfg.drop_last = True

# Augmentation (cutmix/mixup disabled — tracking requires spatial coherence)
cfg.mixup_p = 0.0
cfg.cutmix_p = 0.0
