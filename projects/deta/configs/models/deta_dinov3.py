"""DETA Model Configuration with DINOv3 Backbone.

This module defines the model architecture for DETA (Detection Transformers with
Assignment) using DINOv3 as the backbone.

Architecture Overview:
    1. Backbone: DINOv3 (ViT or ConvNeXt variants)
       - Extracts multi-scale features from intermediate layers
       - Default: ViT-Base/16 with features from layers [3, 7, 11]
    
    2. Neck: ChannelMapper
       - Converts backbone features to 256 channels
       - Generates 5-level feature pyramid for deformable attention
    
    3. Transformer: DeformableDetrTransformer
       - Encoder: 6 layers with multi-scale deformable attention
       - Decoder: 6 layers with iterative bounding box refinement
       - Two-stage: Initial proposals from encoder features
    
    4. Criterion: DETACriterion
       - Focal loss for classification
       - L1 + GIoU loss for bbox regression
       - Assignment for both encoder and decoder predictions

Supported DINOv3 Models:
    - ViT-S/16: embed_dim=384, depth=12
    - ViT-B/16: embed_dim=768, depth=12 (default)
    - ViT-L/16: embed_dim=1024, depth=24
    - ViT-H+/16: embed_dim=1280, depth=32
    - ConvNeXt-Base: dims=[128, 256, 512, 1024]
    - ConvNeXt-Large: dims=[192, 384, 768, 1536]

To customize the backbone:
    1. Set DINOV3_MODEL_NAME to desired variant
    2. Set DINOV3_EMBED_DIM to match the variant's embedding dimension
    3. Set DINOV3_CHECKPOINT to the checkpoint path
    4. Adjust DINOV3_OUT_INDICES based on model depth

Example:
    # For ViT-Large
    DINOV3_MODEL_NAME = "vitl16"
    DINOV3_EMBED_DIM = 1024
    DINOV3_OUT_INDICES = [7, 15, 23]  # 24-layer model
    
    # For ConvNeXt-Base
    DINOV3_MODEL_NAME = "convnext_base"
    DINOV3_OUT_INDICES = [1, 2, 3]  # Use stages 1, 2, 3
    # Note: ConvNeXt uses different channels per stage

Authors: detrex contributors
License: Apache-2.0
"""

import torch.nn as nn

from detectron2.layers import ShapeSpec
from detectron2.config import LazyCall as L

from detrex.modeling.matcher import HungarianMatcher
from detrex.modeling.neck import ChannelMapper
from detrex.layers import PositionEmbeddingSine
from detrex.modeling.backbone import DINOv3Backbone

from projects.deta.modeling import (
    DeformableDETR,
    DeformableDetrTransformerEncoder,
    DeformableDetrTransformerDecoder,
    DeformableDetrTransformer,
    DETACriterion,
)


# DINOv3 ViT-Base configuration
# For different models, adjust the following:
# - model_name: "vits16", "vitb16", "vitl16", "convnext_base", etc.
# - checkpoint_path: Path to local checkpoint
# - out_indices: Layer indices to extract features from
# - neck input_shapes: Match the embed_dim of the backbone

# Model configurations for reference:
# ViT-S: embed_dim=384, depth=12
# ViT-B: embed_dim=768, depth=12
# ViT-L: embed_dim=1024, depth=24
# ViT-H: embed_dim=1280, depth=32
# ConvNeXt-Base: dims=[128, 256, 512, 1024]
# ConvNeXt-Large: dims=[192, 384, 768, 1536]

# Default: ViT-Base with 3 output features from layers [3, 7, 11]
DINOV3_MODEL_NAME = "vitb16"
DINOV3_EMBED_DIM = 768  # ViT-B embed dim
DINOV3_CHECKPOINT = "./dinov3/checkpoints/dinov3_vitb16.pth"
# Extract features from 3 layers (similar to res3, res4, res5 in ResNet)
DINOV3_OUT_INDICES = [3, 7, 11]  # For 12-layer ViT-B
DINOV3_OUT_FEATURES = ["stage1", "stage2", "stage3"]

model = L(DeformableDETR)(
    backbone=L(DINOv3Backbone)(
        model_name=DINOV3_MODEL_NAME,
        pretrained=False,  # We load from checkpoint_path
        checkpoint_path=DINOV3_CHECKPOINT,
        out_features=DINOV3_OUT_FEATURES,
        out_indices=DINOV3_OUT_INDICES,
        freeze_backbone=True,  # Freeze backbone weights
        drop_path_rate=0.0,
    ),
    position_embedding=L(PositionEmbeddingSine)(
        num_pos_feats=128,
        temperature=10000,
        normalize=True,
        offset=-0.5,
    ),
    neck=L(ChannelMapper)(
        input_shapes={
            "stage1": ShapeSpec(channels=DINOV3_EMBED_DIM),
            "stage2": ShapeSpec(channels=DINOV3_EMBED_DIM),
            "stage3": ShapeSpec(channels=DINOV3_EMBED_DIM),
        },
        in_features=DINOV3_OUT_FEATURES,
        out_channels=256,
        num_outs=5,
        kernel_size=1,
        norm_layer=L(nn.GroupNorm)(num_groups=32, num_channels=256),
    ),
    transformer=L(DeformableDetrTransformer)(
        encoder=L(DeformableDetrTransformerEncoder)(
            embed_dim=256,
            num_heads=8,
            feedforward_dim=2048,
            attn_dropout=0.0,
            ffn_dropout=0.0,
            num_layers=6,
            post_norm=False,
            num_feature_levels="${..num_feature_levels}",
        ),
        decoder=L(DeformableDetrTransformerDecoder)(
            embed_dim=256,
            num_heads=8,
            feedforward_dim=2048,
            attn_dropout=0.0,
            ffn_dropout=0.0,
            num_layers=6,
            return_intermediate=True,
            num_feature_levels="${..num_feature_levels}",
        ),
        as_two_stage="${..as_two_stage}",
        num_feature_levels=5,
        two_stage_num_proposals="${..num_queries}",
        assign_first_stage=True,
    ),
    embed_dim=256,
    num_classes=80,
    num_queries=900,
    aux_loss=True,
    with_box_refine=True,
    as_two_stage=True,
    criterion=L(DETACriterion)(
        num_classes=1,
        matcher=L(HungarianMatcher)(
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            cost_class_type="focal_loss_cost",
            alpha=0.25,
            gamma=2.0,
        ),
        weight_dict={
            "loss_class": 1.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
        },
        loss_class_type="focal_loss",
        alpha=0.25,
        gamma=2.0,
        num_queries="${..num_queries}",
        assign_first_stage=True,
        assign_second_stage=True,
    ),
    pixel_mean=[123.675, 116.280, 103.530],
    pixel_std=[58.395, 57.120, 57.375],
    select_box_nums_for_evaluation=300,
    device="cuda",
)

# set aux loss weight dict
if model.aux_loss:
    weight_dict = model.criterion.weight_dict
    aux_weight_dict = {}
    for i in range(model.transformer.decoder.num_layers - 1):
        aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
    aux_weight_dict.update({k + "_enc": v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)
    model.criterion.weight_dict = weight_dict
