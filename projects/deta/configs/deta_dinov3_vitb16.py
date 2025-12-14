"""DETA with DINOv3-ViT-Base/16 Backbone Configuration.

This configuration file sets up DETA (NMS Strikes Back) with a DINOv3 ViT-Base
backbone for COCO object detection.

DINOv3 is a self-supervised vision transformer pretrained on 1.6B+ images,
providing strong visual representations for downstream tasks.

Key Features:
    - Backbone: DINOv3-ViT-Base/16 (768 embed_dim, 12 layers)
    - Multi-scale features: Extracted from layers [3, 7, 11]
    - Frozen backbone: Transfer learning mode (faster convergence)
    - 5-level feature pyramid: Created by ChannelMapper neck

Usage:
    python projects/deta/train_net.py \
        --config-file projects/deta/configs/deta_dinov3_vitb16.py \
        --num-gpus 8

For other DINOv3 variants, modify:
    - model.backbone.model_name (vits16, vitl16, convnext_base, etc.)
    - model.backbone.checkpoint_path
    - model.backbone.out_indices (adjust for model depth)
    - EMBED_DIM and model.neck.input_shapes

See README_DINOV3.md for detailed documentation.
"""

from detrex.config import get_config
from .models.deta_dinov3 import model
from .scheduler.coco_scheduler import lr_multiplier_12ep_10drop as lr_multiplier

# using the default optimizer and dataloader
dataloader = get_config("common/data/coco_detr.py").dataloader
optimizer = get_config("common/optim.py").AdamW
train = get_config("common/train.py").train

# ========== DINOv3 Backbone Configuration ==========
# Specify your DINOv3 checkpoint path here
# Available checkpoints in dinov3/checkpoints:
#   - dinov3_vits16_pretrain_lvd1689m-08c60483.pth (ViT-Small, embed_dim=384)
#   - dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth (ViT-Base, embed_dim=768)
#   - dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth (ViT-Large, embed_dim=1024)
#   - dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth (ViT-Huge+, embed_dim=1280)
#   - dinov3_convnext_base.pth (ConvNeXt-Base, dims=[128,256,512,1024])
#   - dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth (ConvNeXt-Large, dims=[192,384,768,1536])

# Configure backbone - modify these to change the DINOv3 variant
model.backbone.model_name = "vitb16"  # Options: vits16, vitb16, vitl16, vith16plus, convnext_base, convnext_large
model.backbone.checkpoint_path = (
    "./dinov3/checkpoints/dinov3_vitb16.pth"
)
model.backbone.freeze_backbone = (
    True  # Freeze backbone weights (recommended for pretrained)
)

# Out indices for feature extraction (adjust based on model depth):
# ViT-S/B (depth=12): [3, 7, 11] or [2, 5, 8, 11] for 4 scales
# ViT-L (depth=24): [7, 15, 23] or [5, 11, 17, 23] for 4 scales
# ViT-H (depth=32): [9, 19, 31] or [7, 15, 23, 31] for 4 scales
# ConvNeXt: [0, 1, 2, 3] for 4 stages
model.backbone.out_indices = [3, 7, 11]

# Neck configuration - channels should match backbone embed_dim for ViT
# ViT-S: 384, ViT-B: 768, ViT-L: 1024, ViT-H: 1280
# For ConvNeXt, use stage dims: base=[128,256,512,1024], large=[192,384,768,1536]
from detectron2.layers import ShapeSpec
import torch.nn as nn
from detectron2.config import LazyCall as L

EMBED_DIM = 768  # Match your backbone's embed_dim
model.neck.input_shapes = {
    "stage1": ShapeSpec(channels=EMBED_DIM),
    "stage2": ShapeSpec(channels=EMBED_DIM),
    "stage3": ShapeSpec(channels=EMBED_DIM),
}

# ========== Training Configuration ==========
# No init_checkpoint needed - we load DINOv3 weights directly
train.init_checkpoint = ""
train.output_dir = "./output/deta_dinov3_vitb16"

# max training iterations
train.max_iter = 100000
train.eval_period = 200
train.checkpointer.period = 200

# set training devices
train.device = "cuda"
model.device = train.device

# modify dataloader config
dataloader.train.num_workers = 4

# Total batch size (distributed across GPUs)
# E.g., 4 GPUs with batch_size=4 means total_batch_size=16
dataloader.train.total_batch_size = 16

# ========== Optimizer Configuration ==========
# Use lower learning rate for frozen backbone fine-tuning
optimizer.lr = 1e-4
optimizer.weight_decay = 1e-4

# Optionally reduce learning rate for backbone (if not frozen)
# optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1.0
