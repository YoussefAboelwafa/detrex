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
