"""
Feature Assembly Module
=======================

Assemble 6-channel input tensors from RGB frames, optical flow, and masks.

DOCUMENTATION:
- Input Components:
  * RGB frame: (H, W, 3) normalized with ImageNet stats
  * Optical flow: (H, W, 2) masked [Vx, Vy]
  * Vehicle mask: (H, W, 1) binary

- Channel Order (CRITICAL FOR REPRODUCIBILITY):
  * Channel 0: Red (normalized)
  * Channel 1: Green (normalized)
  * Channel 2: Blue (normalized)
  * Channel 3: Optical flow Vx (masked)
  * Channel 4: Optical flow Vy (masked)
  * Channel 5: Vehicle mask (0 or 1)

- Normalization:
  * RGB: ImageNet stats (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  * Flow: Already normalized to [0, 1] in optical_flow.py
  * Mask: Binary (0 or 1)

- Resize: 224x224 using bilinear interpolation (cv2.INTER_LINEAR)
- Order: Resize BEFORE normalization

- Feature Masking:
  * masked_flow = optical_flow * vehicle_mask
  * Only vehicle regions contribute to flow features
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class FeatureAssembler:
    """
    Assemble 6-channel feature tensors for collision prediction.
    
    Args:
        target_size: Output size (height, width), default (224, 224)
        normalize_rgb: Apply ImageNet normalization to RGB
        mask_flow: Multiply flow by vehicle mask
        
    Example:
        >>> assembler = FeatureAssembler()
        >>> tensor = assembler.assemble(rgb_frame, flow, mask)
        >>> print(tensor.shape)  # (6, 224, 224)
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalize_rgb: bool = True,
        mask_flow: bool = True,
    ):
        self.target_size = target_size  # (H, W)
        self.normalize_rgb = normalize_rgb
        self.mask_flow = mask_flow
        
        # Precompute tensors for normalization
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    
    def assemble(
        self,
        rgb: np.ndarray,
        flow: np.ndarray,
        mask: np.ndarray,
    ) -> torch.Tensor:
        """
        Assemble 6-channel feature tensor.
        
        Args:
            rgb: RGB frame (H, W, 3) values in [0, 255] or [0, 1]
            flow: Optical flow (H, W, 2) values in [0, 1] (normalized)
            mask: Vehicle mask (H, W) binary {0, 1}
            
        Returns:
            Tensor (6, 224, 224) with channels [R, G, B, Vx, Vy, Mask]
        """
        h, w = self.target_size
        
        # Step 1: Resize all inputs to target size
        rgb_resized = self._resize_rgb(rgb)      # (H, W, 3)
        flow_resized = self._resize_flow(flow)   # (H, W, 2)
        mask_resized = self._resize_mask(mask)   # (H, W)
        
        # Step 2: Normalize RGB values to [0, 1] if needed
        if rgb_resized.max() > 1.0:
            rgb_resized = rgb_resized / 255.0
        
        # Step 3: Apply feature masking (flow * mask)
        if self.mask_flow:
            mask_expanded = mask_resized[:, :, np.newaxis]  # (H, W, 1)
            flow_masked = flow_resized * mask_expanded       # Element-wise
        else:
            flow_masked = flow_resized
        
        # Step 4: Convert to PyTorch tensors
        rgb_tensor = torch.from_numpy(rgb_resized).permute(2, 0, 1).float()  # (3, H, W)
        flow_tensor = torch.from_numpy(flow_masked).permute(2, 0, 1).float() # (2, H, W)
        mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0).float()    # (1, H, W)
        
        # Step 5: Apply ImageNet normalization to RGB
        if self.normalize_rgb:
            rgb_tensor = (rgb_tensor - self.mean) / self.std
        
        # Step 6: Concatenate to 6-channel tensor
        # Order: [R, G, B, Vx, Vy, Mask]
        tensor = torch.cat([rgb_tensor, flow_tensor, mask_tensor], dim=0)
        
        assert tensor.shape == (6, h, w), f"Expected (6, {h}, {w}), got {tensor.shape}"
        
        return tensor
    
    def assemble_batch(
        self,
        rgb_list: list,
        flow_list: list,
        mask_list: list,
    ) -> torch.Tensor:
        """
        Assemble batch of 6-channel tensors.
        
        Args:
            rgb_list: List of RGB frames
            flow_list: List of optical flow arrays
            mask_list: List of vehicle masks
            
        Returns:
            Tensor (B, 6, 224, 224)
        """
        tensors = []
        for rgb, flow, mask in zip(rgb_list, flow_list, mask_list):
            tensor = self.assemble(rgb, flow, mask)
            tensors.append(tensor)
        
        return torch.stack(tensors)
    
    def _resize_rgb(self, rgb: np.ndarray) -> np.ndarray:
        """Resize RGB frame using bilinear interpolation."""
        return cv2.resize(
            rgb, 
            (self.target_size[1], self.target_size[0]),  # cv2 uses (W, H)
            interpolation=cv2.INTER_LINEAR
        )
    
    def _resize_flow(self, flow: np.ndarray) -> np.ndarray:
        """Resize optical flow using bilinear interpolation."""
        return cv2.resize(
            flow,
            (self.target_size[1], self.target_size[0]),
            interpolation=cv2.INTER_LINEAR
        )
    
    def _resize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Resize mask using nearest neighbor (preserve binary values)."""
        resized = cv2.resize(
            mask.astype(np.float32),
            (self.target_size[1], self.target_size[0]),
            interpolation=cv2.INTER_NEAREST
        )
        return (resized > 0.5).astype(np.float32)
    
    def denormalize_rgb(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert normalized tensor back to displayable RGB image.
        
        Args:
            tensor: Normalized tensor (3, H, W) or (6, H, W)
            
        Returns:
            RGB array (H, W, 3) in [0, 255] uint8
        """
        if tensor.shape[0] == 6:
            tensor = tensor[:3]  # Take RGB channels only
        
        # Denormalize
        rgb = tensor * self.std + self.mean
        
        # Convert to numpy and scale to [0, 255]
        rgb = rgb.permute(1, 2, 0).numpy()
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        
        return rgb
    
    def extract_components(self, tensor: torch.Tensor) -> dict:
        """
        Extract individual components from 6-channel tensor.
        
        Args:
            tensor: Input tensor (6, H, W)
            
        Returns:
            Dictionary with 'rgb', 'flow', 'mask' arrays
        """
        rgb = self.denormalize_rgb(tensor[:3])
        flow = tensor[3:5].permute(1, 2, 0).numpy()
        mask = tensor[5].numpy()
        
        return {
            "rgb": rgb,
            "flow": flow,
            "mask": mask,
        }


def create_6channel_input(
    rgb: np.ndarray,
    flow: np.ndarray,
    mask: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
) -> torch.Tensor:
    """
    Convenience function to create 6-channel input tensor.
    
    Args:
        rgb: RGB frame (H, W, 3)
        flow: Optical flow (H, W, 2)
        mask: Vehicle mask (H, W)
        target_size: Output size
        
    Returns:
        Tensor (6, 224, 224)
    """
    assembler = FeatureAssembler(target_size=target_size)
    return assembler.assemble(rgb, flow, mask)
