"""
Loss Functions Module
=====================

Loss functions for collision prediction training.

DOCUMENTATION:
- BCEWithLogitsLoss: Default, numerically stable binary cross-entropy
- FocalLoss: For handling class imbalance

MODEL OUTPUT FORMAT:
- Models output raw logits (no sigmoid)
- Sigmoid is applied inside the loss function
- This is more numerically stable than BCELoss with external sigmoid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    DOCUMENTATION:
    - Original paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    - Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    - alpha: Balance factor for positive/negative classes
    - gamma: Focusing parameter (reduces loss for well-classified examples)
    
    When gamma=0, reduces to standard cross-entropy.
    Higher gamma focuses more on hard examples.
    
    Args:
        alpha: Balance factor (default: 0.25 for imbalanced datasets)
        gamma: Focusing parameter (default: 2.0)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(
        self, 
        alpha: float = 0.25, 
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self, 
        inputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Raw logits (B, 1) or (B,)
            targets: Ground truth labels (B, 1) or (B,) in {0, 1}
            
        Returns:
            Focal loss value
        """
        # Flatten inputs and targets
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Compute probabilities
        probs = torch.sigmoid(inputs)
        
        # Compute focal weights
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute alpha weights
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        
        # Apply focal and alpha weights
        focal_loss = alpha_t * focal_weight * bce_loss
        
        # Reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingBCELoss(nn.Module):
    """
    Binary cross-entropy with label smoothing.
    
    DOCUMENTATION:
    - Converts hard labels {0, 1} to soft labels {epsilon, 1-epsilon}
    - Helps prevent overconfident predictions
    - Used in V2 model variants
    
    Args:
        smoothing: Smoothing factor (default: 0.1)
            - Label 0 becomes epsilon/2
            - Label 1 becomes 1 - epsilon/2
    """
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(
        self, 
        inputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute label-smoothed BCE loss.
        
        Args:
            inputs: Raw logits (B, 1) or (B,)
            targets: Ground truth labels (B, 1) or (B,)
            
        Returns:
            Loss value
        """
        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Apply label smoothing
        smooth_targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        
        # Compute loss
        loss = F.binary_cross_entropy_with_logits(inputs, smooth_targets)
        
        return loss


def get_loss_function(
    loss_type: str = "bce",
    pos_weight: float = 1.0,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> nn.Module:
    """
    Get loss function by name.
    
    Args:
        loss_type: 'bce', 'focal', or 'smooth_bce'
        pos_weight: Weight for positive class (BCEWithLogitsLoss)
        focal_alpha: Alpha for focal loss
        focal_gamma: Gamma for focal loss
        label_smoothing: Smoothing factor for smooth_bce
        
    Returns:
        Loss function module
    """
    if loss_type == "bce":
        weight = torch.tensor([pos_weight]) if pos_weight != 1.0 else None
        return nn.BCEWithLogitsLoss(pos_weight=weight)
    
    elif loss_type == "focal":
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    
    elif loss_type == "smooth_bce":
        return LabelSmoothingBCELoss(smoothing=label_smoothing)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
