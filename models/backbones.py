"""
Backbone Model Implementations
==============================

All 6 model architectures for collision detection.

MODEL DOCUMENTATION:

1. FastViT-T8 (Apple)
   - Library: timm.create_model('fastvit_t8')
   - Type: Hybrid Vision Transformer
   - Parameters: 3.26M
   - Feature dim: 384
   - Pretrained: ImageNet-1k

2. FastViT-T12 (Apple)
   - Library: timm.create_model('fastvit_t12')
   - Type: Hybrid Vision Transformer
   - Parameters: 6.8M
   - Feature dim: 384
   - Pretrained: ImageNet-1k

3. EfficientNet-B0
   - Library: torchvision.models.efficientnet_b0
   - Type: Compound-scaled CNN
   - Parameters: 4.01M
   - Feature dim: 1280
   - Pretrained: ImageNet-1k

4. MobileNetV3-Small
   - Library: torchvision.models.mobilenet_v3_small
   - Type: Lightweight Mobile CNN
   - Parameters: 0.93M
   - Feature dim: 576
   - SE blocks: Yes

5. ConvNeXt-Tiny
   - Library: torchvision.models.convnext_tiny
   - Type: Modern CNN (ViT-inspired)
   - Parameters: 28.6M
   - Feature dim: 768
   - Pretrained: ImageNet-1k

6. ResNet-18
   - Library: torchvision.models.resnet18
   - Type: Classic Residual CNN
   - Parameters: 11.19M
   - Feature dim: 512
   - Pretrained: ImageNet-1k
"""

import logging

import torch
import torch.nn as nn
from torchvision import models

from .base import BaseCollisionDetector

logger = logging.getLogger(__name__)

# Check timm availability
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    logger.warning("timm not installed. FastViT models unavailable.")


class FastViTCollisionDetector(BaseCollisionDetector):
    """
    FastViT-based collision detector.
    
    FastViT is a hybrid vision transformer from Apple that combines
    CNN efficiency with transformer expressiveness.
    
    Args:
        variant: 't8' or 't12'
        pretrained: Use ImageNet pretrained weights
        dropout: Classification head dropout
        num_frames: Temporal frames
    """
    
    def __init__(
        self,
        variant: str = "t8",
        pretrained: bool = True,
        dropout: float = 0.35,
        num_frames: int = 3,
    ):
        if not TIMM_AVAILABLE:
            raise ImportError("timm required for FastViT. Install: pip install timm")
        
        model_name = f"fastvit_{variant}"
        backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            feat_dim = backbone(dummy).shape[1]
        
        super().__init__(
            backbone=backbone,
            feat_dim=feat_dim,
            dropout=dropout,
            num_frames=num_frames,
        )
        
        logger.info(f"Created FastViT-{variant.upper()}: feat_dim={feat_dim}")


class EfficientNetB0CollisionDetector(BaseCollisionDetector):
    """
    EfficientNet-B0 collision detector.
    
    EfficientNet uses compound scaling to balance depth, width, and resolution.
    B0 is the baseline model with good accuracy/efficiency tradeoff.
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.35,
        num_frames: int = 3,
    ):
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        
        # Get feature dimension before removing classifier
        feat_dim = backbone.classifier[1].in_features  # 1280
        
        # Remove classifier
        backbone.classifier = nn.Identity()
        
        super().__init__(
            backbone=backbone,
            feat_dim=feat_dim,
            dropout=dropout,
            num_frames=num_frames,
        )
        
        logger.info(f"Created EfficientNet-B0: feat_dim={feat_dim}")


class MobileNetV3CollisionDetector(BaseCollisionDetector):
    """
    MobileNetV3-Small collision detector.
    
    Ultra-lightweight model designed for mobile/edge deployment.
    Uses SE (Squeeze-and-Excitation) blocks for channel attention.
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.30,
        num_frames: int = 3,
    ):
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)
        
        # Get feature dimension
        feat_dim = backbone.classifier[0].in_features  # 576
        
        # Remove classifier
        backbone.classifier = nn.Identity()
        
        super().__init__(
            backbone=backbone,
            feat_dim=feat_dim,
            dropout=dropout,
            num_frames=num_frames,
        )
        
        logger.info(f"Created MobileNetV3-Small: feat_dim={feat_dim}")


class ConvNeXtTinyCollisionDetector(BaseCollisionDetector):
    """
    ConvNeXt-Tiny collision detector.
    
    Modern pure CNN architecture inspired by Vision Transformers.
    Larger model with higher accuracy but more compute.
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.40,
        num_frames: int = 3,
    ):
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        backbone = models.convnext_tiny(weights=weights)
        
        # Get feature dimension
        feat_dim = backbone.classifier[2].in_features  # 768
        
        # Remove classifier
        backbone.classifier = nn.Identity()
        
        super().__init__(
            backbone=backbone,
            feat_dim=feat_dim,
            dropout=dropout,
            num_frames=num_frames,
        )
        
        logger.info(f"Created ConvNeXt-Tiny: feat_dim={feat_dim}")


class ResNet18CollisionDetector(BaseCollisionDetector):
    """
    ResNet-18 collision detector.
    
    Classic residual network serving as baseline.
    Good balance of simplicity and performance.
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.35,
        num_frames: int = 3,
    ):
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        
        # Get feature dimension
        feat_dim = backbone.fc.in_features  # 512
        
        # Remove classifier
        backbone.fc = nn.Identity()
        
        super().__init__(
            backbone=backbone,
            feat_dim=feat_dim,
            dropout=dropout,
            num_frames=num_frames,
        )
        
        logger.info(f"Created ResNet-18: feat_dim={feat_dim}")


# Model registry for factory function
MODEL_CLASSES = {
    "fastvit_t8": FastViTCollisionDetector,
    "fastvit_t12": FastViTCollisionDetector,
    "efficientnet_b0": EfficientNetB0CollisionDetector,
    "mobilenetv3_small": MobileNetV3CollisionDetector,
    "convnext_tiny": ConvNeXtTinyCollisionDetector,
    "resnet18": ResNet18CollisionDetector,
}


def get_backbone_class(architecture: str):
    """Get model class for architecture name."""
    arch_lower = architecture.lower().replace("-", "_").replace(" ", "_")
    
    if arch_lower not in MODEL_CLASSES:
        available = list(MODEL_CLASSES.keys())
        raise ValueError(f"Unknown architecture: {architecture}. Available: {available}")
    
    return MODEL_CLASSES[arch_lower]
