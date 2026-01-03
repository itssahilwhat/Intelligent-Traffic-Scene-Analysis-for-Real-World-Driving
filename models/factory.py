"""
Model Factory Module
====================

Factory functions for creating collision detection models.
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def create_model(
    model_name: str,
    pretrained: bool = True,
    dropout: Optional[float] = None,
    num_frames: int = 3,
    checkpoint_path: Optional[str] = None,
):
    """
    Create a collision detection model by name.
    
    Args:
        model_name: Model name (e.g., "EfficientNet-B0", "FastViT-T8")
        pretrained: Use ImageNet pretrained weights
        dropout: Override default dropout rate
        num_frames: Number of temporal frames
        checkpoint_path: Optional path to load trained weights
        
    Returns:
        Collision detection model instance
        
    Example:
        >>> model = create_model("EfficientNet-B0")
        >>> model = create_model("FastViT-T8", dropout=0.4)
    """
    from .backbones import (
        FastViTCollisionDetector,
        EfficientNetB0CollisionDetector,
        MobileNetV3CollisionDetector,
        ConvNeXtTinyCollisionDetector,
        ResNet18CollisionDetector,
    )
    
    # Normalize name
    name_lower = model_name.lower().replace("-", "").replace("_", "").replace(" ", "")
    
    # Model mapping
    model_map = {
        "fastvitt8": (FastViTCollisionDetector, {"variant": "t8", "dropout": 0.35}),
        "fastvitt12": (FastViTCollisionDetector, {"variant": "t12", "dropout": 0.35}),
        "efficientnetb0": (EfficientNetB0CollisionDetector, {"dropout": 0.35}),
        "mobilenetv3small": (MobileNetV3CollisionDetector, {"dropout": 0.30}),
        "convnexttiny": (ConvNeXtTinyCollisionDetector, {"dropout": 0.40}),
        "resnet18": (ResNet18CollisionDetector, {"dropout": 0.35}),
    }
    
    if name_lower not in model_map:
        available = list(model_map.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")
    
    model_class, default_kwargs = model_map[name_lower]
    
    # Override defaults
    kwargs = {
        "pretrained": pretrained,
        "num_frames": num_frames,
        **default_kwargs,
    }
    
    if dropout is not None:
        kwargs["dropout"] = dropout
    
    # Create model
    model = model_class(**kwargs)
    
    # Load checkpoint if specified
    if checkpoint_path:
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu")
        
        # Handle different checkpoint formats
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        elif "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        else:
            model.load_state_dict(state)
    
    return model


def list_models() -> None:
    """Print available models with their specifications."""
    models = {
        "FastViT-T8": {
            "architecture": "Hybrid Vision Transformer",
            "params": "3.26M",
            "feat_dim": 384,
            "default_dropout": 0.35,
        },
        "FastViT-T12": {
            "architecture": "Hybrid Vision Transformer",
            "params": "6.8M",
            "feat_dim": 384,
            "default_dropout": 0.35,
        },
        "EfficientNet-B0": {
            "architecture": "Compound-scaled CNN",
            "params": "4.01M",
            "feat_dim": 1280,
            "default_dropout": 0.35,
        },
        "MobileNetV3-Small": {
            "architecture": "Lightweight Mobile CNN",
            "params": "0.93M",
            "feat_dim": 576,
            "default_dropout": 0.30,
        },
        "ConvNeXt-Tiny": {
            "architecture": "Modern CNN",
            "params": "28.6M",
            "feat_dim": 768,
            "default_dropout": 0.40,
        },
        "ResNet-18": {
            "architecture": "Residual CNN",
            "params": "11.19M",
            "feat_dim": 512,
            "default_dropout": 0.35,
        },
    }
    
    print("\n" + "=" * 70)
    print("AVAILABLE MODELS")
    print("=" * 70)
    
    for name, spec in models.items():
        print(f"\n{name}")
        print(f"  Architecture: {spec['architecture']}")
        print(f"  Parameters:   {spec['params']}")
        print(f"  Feature Dim:  {spec['feat_dim']}")
        print(f"  Dropout:      {spec['default_dropout']}")
    
    print("\n" + "=" * 70)


def get_model_info(model_name: str) -> dict:
    """Get model specifications as dictionary."""
    models = {
        "FastViT-T8": {"params": "3.26M", "feat_dim": 384},
        "FastViT-T12": {"params": "6.8M", "feat_dim": 384},
        "EfficientNet-B0": {"params": "4.01M", "feat_dim": 1280},
        "MobileNetV3-Small": {"params": "0.93M", "feat_dim": 576},
        "ConvNeXt-Tiny": {"params": "28.6M", "feat_dim": 768},
        "ResNet-18": {"params": "11.19M", "feat_dim": 512},
    }
    
    name_normalized = model_name.replace("_", "-")
    for key in models:
        if key.lower() == name_normalized.lower():
            return models[key]
    
    return {}
