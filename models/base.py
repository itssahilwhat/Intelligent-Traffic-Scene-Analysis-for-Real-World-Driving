"""
Base Collision Detector Module
==============================

Base class for all collision detection models.

ARCHITECTURE DOCUMENTATION:
- Input: 6-channel tensor (B, 6, 224, 224) with [RGB, Flow_x, Flow_y, Mask]
- Channel Projection: 6→3 using learned 1x1 convolution
- Backbone: Any CNN/ViT feature extractor
- Temporal Aggregation: Attention over multiple frames
- Head: FC(feat_dim→256)→ReLU→Dropout→FC(256→128)→ReLU→Dropout→FC(128→1)
- Output: Raw logits (sigmoid applied in loss function)

HEAD ARCHITECTURE RATIONALE:
- 2-layer MLP with gradual dimension reduction
- Dropout for regularization
- No sigmoid (using BCEWithLogitsLoss for numerical stability)
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ChannelProjection(nn.Module):
    """
    Project 6-channel input to 3-channel for pretrained backbones.
    
    DOCUMENTATION:
    - Method: Learned 1x1 convolution
    - Initialization: Weights initialized to copy RGB and average Flow/Mask
    - Alternative considered: Copying RGB weights (less flexible)
    """
    
    def __init__(self, in_channels: int = 6, out_channels: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize to pass RGB through and average the rest."""
        with torch.no_grad():
            self.conv.weight.zero_()
            # RGB channels: identity mapping
            self.conv.weight[0, 0] = 1.0  # R -> R
            self.conv.weight[1, 1] = 1.0  # G -> G
            self.conv.weight[2, 2] = 1.0  # B -> B
            # Flow and mask: small contribution
            self.conv.weight[0, 3:] = 0.1  # Flow/Mask -> R
            self.conv.weight[1, 3:] = 0.1  # Flow/Mask -> G
            self.conv.weight[2, 3:] = 0.1  # Flow/Mask -> B
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TemporalAggregation(nn.Module):
    """
    Aggregate features across multiple frames using attention.
    
    DOCUMENTATION:
    - Input: (batch, num_frames, feat_dim)
    - Method: Learned attention weights
    - Output: (batch, feat_dim)
    """
    
    def __init__(self, feat_dim: int, method: str = "attention"):
        super().__init__()
        self.method = method
        
        if method == "attention":
            self.attention = nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 4),
                nn.Tanh(),
                nn.Linear(feat_dim // 4, 1),
            )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_frames, feat_dim)
            
        Returns:
            (batch, feat_dim)
        """
        if self.method == "mean":
            return x.mean(dim=1)
        
        # Attention-weighted aggregation
        weights = self.attention(x)  # (batch, num_frames, 1)
        weights = torch.softmax(weights, dim=1)
        
        return (x * weights).sum(dim=1)


class ClassificationHead(nn.Module):
    """
    Classification head for binary collision prediction.
    
    ARCHITECTURE:
    - FC1: feat_dim → 256, ReLU, Dropout
    - FC2: 256 → 128, ReLU, Dropout
    - FC3: 128 → 1 (logits, no activation)
    """
    
    def __init__(self, feat_dim: int, dropout: float = 0.35):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(128, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class BaseCollisionDetector(nn.Module):
    """
    Base class for collision detection models.
    
    Features:
    - Channel projection (6→3) for pretrained backbones
    - Temporal aggregation across multiple frames
    - Classification head with dropout
    
    Args:
        backbone: Feature extraction network
        feat_dim: Output dimension of backbone
        input_channels: Input channels (default: 6)
        dropout: Dropout rate (default: 0.35)
        num_frames: Number of temporal frames (default: 3)
        
    Example:
        >>> class MyModel(BaseCollisionDetector):
        ...     def __init__(self):
        ...         backbone = resnet18(pretrained=True)
        ...         super().__init__(backbone, feat_dim=512)
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        feat_dim: int,
        input_channels: int = 6,
        dropout: float = 0.35,
        num_frames: int = 3,
    ):
        super().__init__()
        
        self.feat_dim = feat_dim
        self.num_frames = num_frames
        
        # Channel projection (6 → 3)
        self.channel_proj = ChannelProjection(input_channels, 3)
        
        # Backbone (feature extractor)
        self.backbone = backbone
        
        # Temporal aggregation
        self.temporal_agg = TemporalAggregation(feat_dim, method="attention")
        
        # Classification head
        self.classifier = ClassificationHead(feat_dim, dropout)
        
        logger.info(f"Created model: feat_dim={feat_dim}, dropout={dropout}")
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input. Override for custom backbones.
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            Features (B, feat_dim)
        """
        return self.backbone(x)
    
    def forward_single(
        self, 
        frame: torch.Tensor, 
        flow: torch.Tensor
    ) -> torch.Tensor:
        """
        Process a single frame-flow pair.
        
        Args:
            frame: RGB frame (B, 3, H, W)
            flow: Optical flow (B, 3, H, W)
            
        Returns:
            Features (B, feat_dim)
        """
        # Concatenate frame and flow
        x = torch.cat([frame, flow], dim=1)  # (B, 6, H, W)
        
        # Project to 3 channels
        x = self.channel_proj(x)  # (B, 3, H, W)
        
        # Extract features
        features = self.extract_features(x)  # (B, feat_dim)
        
        return features
    
    def forward(
        self, 
        frames: torch.Tensor, 
        flows: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for collision detection.
        
        Args:
            frames: RGB frames (B, T, 3, H, W) or (B, 3, H, W)
            flows: Optical flow (B, T, 3, H, W) or (B, 3, H, W)
            
        Returns:
            Logits (B, 1)
        """
        # Handle single-frame input
        if frames.dim() == 4:
            frames = frames.unsqueeze(1)  # (B, 1, 3, H, W)
            flows = flows.unsqueeze(1)
        
        batch_size, num_frames = frames.shape[:2]
        
        # Process each frame
        all_features = []
        for t in range(num_frames):
            feat = self.forward_single(frames[:, t], flows[:, t])
            all_features.append(feat)
        
        # Stack: (B, T, feat_dim)
        features = torch.stack(all_features, dim=1)
        
        # Temporal aggregation: (B, feat_dim)
        aggregated = self.temporal_agg(features)
        
        # Classification: (B, 1)
        logits = self.classifier(aggregated)
        
        return logits
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_backbone(self) -> None:
        """Freeze backbone parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen")
    
    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen")
