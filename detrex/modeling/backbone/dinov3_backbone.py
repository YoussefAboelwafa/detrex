# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DINOv3 Backbone wrapper for detrex/detectron2.

This module provides a wrapper around DINOv3 models (ViT and ConvNeXt variants)
to be used as backbones in detection models like DETA.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling.backbone import Backbone
from detectron2.layers import ShapeSpec

logger = logging.getLogger(__name__)

# Add dinov3 to path
DINOV3_ROOT = Path(__file__).parent.parent.parent.parent / "dinov3"
if str(DINOV3_ROOT) not in sys.path:
    sys.path.insert(0, str(DINOV3_ROOT))

__all__ = ["DINOv3Backbone", "DINOv3SimpleFeaturePyramid"]


class DINOv3Backbone(Backbone):
    """
    A wrapper for DINOv3 models (ViT or ConvNeXt) to use as a detectron2 backbone.

    This backbone extracts multi-scale features from intermediate layers of the
    DINOv3 model, making them compatible with detection heads like DETA/DINO.
    """

    # Model configurations
    VIT_CONFIGS = {
        "vits16": {"embed_dim": 384, "depth": 12, "patch_size": 16},
        "vits16plus": {"embed_dim": 384, "depth": 12, "patch_size": 16},
        "vitb16": {"embed_dim": 768, "depth": 12, "patch_size": 16},
        "vitl16": {"embed_dim": 1024, "depth": 24, "patch_size": 16},
        "vitl16plus": {"embed_dim": 1024, "depth": 24, "patch_size": 16},
        "vith16plus": {"embed_dim": 1280, "depth": 32, "patch_size": 16},
        "vit7b16": {"embed_dim": 4096, "depth": 40, "patch_size": 16},
    }

    CONVNEXT_CONFIGS = {
        "convnext_tiny": {"dims": [96, 192, 384, 768]},
        "convnext_small": {"dims": [96, 192, 384, 768]},
        "convnext_base": {"dims": [128, 256, 512, 1024]},
        "convnext_large": {"dims": [192, 384, 768, 1536]},
    }

    def __init__(
        self,
        model_name: str = "vitb16",
        pretrained: bool = True,
        checkpoint_path: Optional[str] = None,
        out_features: List[str] = ["stage1", "stage2", "stage3"],
        out_indices: List[int] = [0, 1, 2],  # Which intermediate layers to output
        freeze_backbone: bool = True,
        drop_path_rate: float = 0.0,
    ):
        """
        Args:
            model_name: Name of the DINOv3 model variant. Options:
                ViT: "vits16", "vits16plus", "vitb16", "vitl16", "vitl16plus", "vith16plus", "vit7b16"
                ConvNeXt: "convnext_tiny", "convnext_small", "convnext_base", "convnext_large"
            pretrained: Whether to load pretrained weights from DINOv3.
            checkpoint_path: Path to local checkpoint file. If provided, loads from this path
                instead of downloading pretrained weights.
            out_features: Names for the output feature maps.
            out_indices: Indices of layers to extract features from.
                For ViT: indices into the blocks (e.g., [3, 7, 11] for vitb16)
                For ConvNeXt: indices of stages (0, 1, 2, 3)
            freeze_backbone: Whether to freeze backbone weights.
            drop_path_rate: Drop path rate for training.
        """
        super().__init__()

        self.model_name = model_name
        self.out_features = out_features
        self.out_indices = out_indices
        self.freeze_backbone = freeze_backbone

        # Determine model type
        self.is_vit = model_name in self.VIT_CONFIGS
        self.is_convnext = model_name in self.CONVNEXT_CONFIGS

        if not self.is_vit and not self.is_convnext:
            raise ValueError(
                f"Unknown model_name: {model_name}. "
                f"Supported ViT: {list(self.VIT_CONFIGS.keys())}, "
                f"Supported ConvNeXt: {list(self.CONVNEXT_CONFIGS.keys())}"
            )

        # Build the model
        self.model = self._build_model(
            model_name, pretrained, checkpoint_path, drop_path_rate
        )

        # Get output channel info
        if self.is_vit:
            config = self.VIT_CONFIGS[model_name]
            self.embed_dim = config["embed_dim"]
            self.patch_size = config["patch_size"]
            self._out_feature_channels = {name: self.embed_dim for name in out_features}
            # For ViT, all outputs have the same stride (patch_size)
            self._out_feature_strides = {name: self.patch_size for name in out_features}
        else:
            config = self.CONVNEXT_CONFIGS[model_name]
            dims = config["dims"]
            # ConvNeXt has 4 stages with strides [4, 8, 16, 32]
            strides = [4, 8, 16, 32]
            self._out_feature_channels = {
                name: dims[idx] for name, idx in zip(out_features, out_indices)
            }
            self._out_feature_strides = {
                name: strides[idx] for name, idx in zip(out_features, out_indices)
            }

        # Freeze if requested
        if freeze_backbone:
            self._freeze()

    def _build_model(
        self,
        model_name: str,
        pretrained: bool,
        checkpoint_path: Optional[str],
        drop_path_rate: float,
    ):
        """Build the DINOv3 model."""
        # Ensure dinov3 is in path
        if str(DINOV3_ROOT) not in sys.path:
            sys.path.insert(0, str(DINOV3_ROOT))

        from dinov3.hub import backbones

        # Get the model builder function
        builder_name = f"dinov3_{model_name}"
        if not hasattr(backbones, builder_name):
            raise ValueError(f"DINOv3 does not have model: {builder_name}")

        builder_fn = getattr(backbones, builder_name)

        # Build model
        # Note: DINOv3 builder functions have drop_path_rate hardcoded internally,
        # so we don't pass it as a kwarg to avoid "multiple values" error
        if checkpoint_path is not None:
            # Resolve checkpoint path
            ckpt_path = Path(checkpoint_path)
            if not ckpt_path.is_absolute():
                # Try relative to workspace root
                ckpt_path = DINOV3_ROOT.parent / checkpoint_path

            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

            logger.info(f"Loading DINOv3 {model_name} from checkpoint: {ckpt_path}")
            model = builder_fn(pretrained=False)
            state_dict = torch.load(
                str(ckpt_path), map_location="cpu", weights_only=True
            )
            # Handle different checkpoint formats
            if "model" in state_dict:
                state_dict = state_dict["model"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict, strict=True)
            logger.info(f"Successfully loaded DINOv3 {model_name} weights")
        else:
            logger.info(f"Loading DINOv3 {model_name} pretrained={pretrained}")
            model = builder_fn(pretrained=pretrained)

        return model

    def _freeze(self):
        """Freeze all backbone parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
        logger.info(f"Froze DINOv3 {self.model_name} backbone weights")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass that extracts multi-scale features.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Dictionary mapping feature names to feature tensors.
            For ViT: features are reshaped to (B, C, H', W')
            For ConvNeXt: features are already (B, C, H', W')
        """
        if self.is_vit:
            return self._forward_vit(x)
        else:
            return self._forward_convnext(x)

    def _forward_vit(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for ViT models."""
        B, C, H, W = x.shape

        # Get intermediate layer outputs
        # The model.get_intermediate_layers returns features from specified layers
        outputs = self.model.get_intermediate_layers(
            x,
            n=self.out_indices,
            reshape=True,  # Reshape to (B, C, H', W')
            return_class_token=False,
            norm=True,
        )

        # Build output dictionary
        out_dict = {}
        for name, feat in zip(self.out_features, outputs):
            out_dict[name] = feat

        return out_dict

    def _forward_convnext(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for ConvNeXt models."""
        # Get intermediate layer outputs
        outputs = self.model.get_intermediate_layers(
            x,
            n=self.out_indices,
            reshape=True,  # Keep as (B, C, H, W)
            return_class_token=False,
            norm=True,
        )

        # Build output dictionary
        out_dict = {}
        for name, feat in zip(self.out_features, outputs):
            out_dict[name] = feat

        return out_dict

    def output_shape(self) -> Dict[str, ShapeSpec]:
        """Return the output shape specification for each feature."""
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self.out_features
        }


class DINOv3SimpleFeaturePyramid(Backbone):
    """
    A Simple Feature Pyramid Network on top of DINOv3 backbone.

    This is similar to the SimpleFeaturePyramid used with EVA/ViTDet,
    creating multi-scale feature maps from DINOv3 single-scale ViT output.
    """

    def __init__(
        self,
        backbone: DINOv3Backbone,
        in_feature: str,
        out_channels: int = 256,
        scale_factors: List[float] = [4.0, 2.0, 1.0, 0.5],
        norm: str = "LN",
        out_features: List[str] = ["p3", "p4", "p5", "p6"],
    ):
        """
        Args:
            backbone: DINOv3Backbone instance.
            in_feature: Name of input feature from backbone.
            out_channels: Number of output channels for all pyramid levels.
            scale_factors: Scale factors for creating multi-scale features.
                > 1.0 means upsampling, < 1.0 means downsampling.
            norm: Normalization type ("LN" for LayerNorm, "GN" for GroupNorm).
            out_features: Names for output feature maps.
        """
        super().__init__()

        self.backbone = backbone
        self.in_feature = in_feature
        self.scale_factors = scale_factors
        self._out_features = out_features

        # Get input channels from backbone
        in_channels = backbone._out_feature_channels[in_feature]
        in_stride = backbone._out_feature_strides[in_feature]

        # Build lateral and output convolutions for each scale
        self.stages = nn.ModuleList()
        self._out_feature_strides = {}
        self._out_feature_channels = {}

        for idx, (scale, out_name) in enumerate(zip(scale_factors, out_features)):
            out_stride = int(in_stride / scale)
            self._out_feature_strides[out_name] = out_stride
            self._out_feature_channels[out_name] = out_channels

            # Create the stage
            if scale == 4.0:
                # 4x upsampling
                layers = [
                    nn.ConvTranspose2d(
                        in_channels, in_channels // 2, kernel_size=2, stride=2
                    ),
                    self._get_norm(norm, in_channels // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(
                        in_channels // 2, in_channels // 4, kernel_size=2, stride=2
                    ),
                    self._get_norm(norm, in_channels // 4),
                    nn.GELU(),
                    nn.Conv2d(in_channels // 4, out_channels, kernel_size=1),
                ]
            elif scale == 2.0:
                # 2x upsampling
                layers = [
                    nn.ConvTranspose2d(
                        in_channels, in_channels // 2, kernel_size=2, stride=2
                    ),
                    self._get_norm(norm, in_channels // 2),
                    nn.GELU(),
                    nn.Conv2d(in_channels // 2, out_channels, kernel_size=1),
                ]
            elif scale == 1.0:
                # No scaling
                layers = [
                    nn.Conv2d(in_channels, out_channels, kernel_size=1),
                    self._get_norm(norm, out_channels),
                ]
            elif scale == 0.5:
                # 2x downsampling
                layers = [
                    nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
                    self._get_norm(norm, out_channels),
                ]
            elif scale == 0.25:
                # 4x downsampling
                layers = [
                    nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
                    self._get_norm(norm, out_channels),
                    nn.GELU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2),
                    self._get_norm(norm, out_channels),
                ]
            else:
                raise ValueError(f"Unsupported scale factor: {scale}")

            self.stages.append(nn.Sequential(*layers))

        # Initialize weights
        self._init_weights()

    def _get_norm(self, norm: str, channels: int) -> nn.Module:
        """Get normalization layer."""
        if norm == "LN":
            return nn.GroupNorm(1, channels)  # LayerNorm equivalent for conv
        elif norm == "GN":
            return nn.GroupNorm(32, channels)
        elif norm == "BN":
            return nn.BatchNorm2d(channels)
        else:
            return nn.Identity()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Dictionary mapping feature names to multi-scale feature tensors.
        """
        # Get backbone features
        backbone_features = self.backbone(x)
        feat = backbone_features[self.in_feature]

        # Apply FPN stages
        outputs = {}
        for stage, out_name in zip(self.stages, self._out_features):
            outputs[out_name] = stage(feat)

        return outputs

    def output_shape(self) -> Dict[str, ShapeSpec]:
        """Return the output shape specification for each feature."""
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }
