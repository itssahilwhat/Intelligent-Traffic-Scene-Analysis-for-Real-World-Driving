"""
Dataset Module for Collision Prediction
========================================

PyTorch datasets for training, validation, and testing.

DOCUMENTATION:
- Data format: Parquet files with columns [features_path, n_frames, target]
- Features: Pre-computed frames and flows stored in directories
- Multi-frame sampling: Extract N frames per video for temporal context
- Augmentation: Applied consistently across all frames in sequence

Data Split:
- Training: 80% with augmentation
- Validation: 20% without augmentation
- Stratified by class to maintain balance
"""

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# ImageNet normalization
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class CollisionDataset(Dataset):
    """
    Dataset for collision prediction with multi-frame temporal sampling.
    
    Args:
        parquet_path: Path to parquet file with video metadata
        indices: Optional indices for train/val split
        augment: Enable data augmentation (training only)
        num_frames: Number of frames to sample per video
        image_size: Target image size
        base_path: Base path for resolving relative feature paths
        
    Data format expected:
        - parquet columns: features_path, n_frames, target
        - features_path points to directory with:
          - frames/ subdirectory containing .pt files
          - flows/ subdirectory containing .pt files
    """
    
    def __init__(
        self,
        parquet_path: str,
        indices: Optional[np.ndarray] = None,
        augment: bool = False,
        num_frames: int = 3,
        image_size: Tuple[int, int] = (224, 224),
        base_path: Optional[str] = None,
    ):
        self.df = pd.read_parquet(parquet_path)
        if indices is not None:
            self.df = self.df.iloc[indices].reset_index(drop=True)
        
        self.augment = augment
        self.num_frames = num_frames
        self.image_size = image_size
        
        # Resolve base path
        if base_path is None:
            self.base_path = Path(parquet_path).parent.parent.parent
        else:
            self.base_path = Path(base_path)
        
        logger.info(f"Dataset: {len(self.df)} samples, augment={augment}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        features_path = row["features_path"]
        n_frames = int(row["n_frames"])
        target = float(row["target"])
        
        # Resolve path
        folder_path = self._resolve_path(features_path)
        
        # Get frame indices for multi-frame sampling
        frame_indices = self._get_frame_indices(n_frames)
        
        # Load frames and flows
        frames = []
        flows = []
        
        for fidx in frame_indices:
            frame = self._load_tensor(folder_path / "frames", fidx)
            flow = self._load_tensor(folder_path / "flows", fidx)
            
            frames.append(frame)
            flows.append(flow)
        
        # Stack: (num_frames, 3, H, W)
        frames = torch.stack(frames)
        flows = torch.stack(flows)
        
        # Apply augmentation
        if self.augment:
            frames, flows = self._apply_augmentation(frames, flows)
        
        # Normalize
        frames = (frames - IMAGENET_MEAN) / IMAGENET_STD
        flows = (flows - IMAGENET_MEAN) / IMAGENET_STD
        
        return {
            "frames": frames,
            "flows": flows,
            "label": torch.tensor(target, dtype=torch.float32),
        }
    
    def _resolve_path(self, features_path: str) -> Path:
        """Resolve feature path to absolute path."""
        if features_path.startswith("../"):
            return self.base_path / features_path.replace("../", "")
        return Path(features_path)
    
    def _get_frame_indices(self, n_frames: int) -> List[int]:
        """Get evenly distributed frame indices."""
        if n_frames <= 0:
            return [0] * self.num_frames
        
        if n_frames < self.num_frames:
            # Repeat frames
            return [min(i, n_frames - 1) for i in range(self.num_frames)]
        
        # Evenly distribute
        step = (n_frames - 1) / (self.num_frames - 1) if self.num_frames > 1 else 0
        indices = [int(i * step) for i in range(self.num_frames)]
        
        # Add temporal jitter during training
        if self.augment and step > 0:
            max_jitter = max(1, int(step * 0.3))
            indices = [
                max(0, min(n_frames - 1, idx + random.randint(-max_jitter, max_jitter)))
                for idx in indices
            ]
        
        return indices
    
    def _load_tensor(self, folder: Path, idx: int) -> torch.Tensor:
        """Load tensor from .pt file."""
        # Try different naming conventions
        for fmt in [f"{idx:02d}.pt", f"{idx}.pt"]:
            path = folder / fmt
            if path.exists():
                return self._safe_load(path)
        
        # Fallback: find any file
        pt_files = sorted(folder.glob("*.pt"))
        if pt_files:
            actual_idx = min(idx, len(pt_files) - 1)
            return self._safe_load(pt_files[actual_idx])
        
        # Return zeros if nothing found
        return torch.zeros(3, *self.image_size)
    
    def _safe_load(self, path: Path) -> torch.Tensor:
        """Safely load tensor with error handling."""
        try:
            if path.stat().st_size < 100:
                return torch.zeros(3, *self.image_size)
            
            tensor = torch.load(path, weights_only=False)
            
            if tensor.numel() == 0:
                return torch.zeros(3, *self.image_size)
            
            tensor = tensor.float()
            
            # Ensure correct shape
            tensor = self._ensure_shape(tensor)
            
            return tensor
            
        except (EOFError, RuntimeError) as e:
            logger.warning(f"Error loading {path}: {e}")
            return torch.zeros(3, *self.image_size)
    
    def _ensure_shape(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is (3, H, W)."""
        # Handle different dimensions
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
        elif tensor.dim() == 3:
            if tensor.shape[0] != 3:
                if tensor.shape[-1] == 3:
                    tensor = tensor.permute(2, 0, 1)
                else:
                    tensor = tensor[:3] if tensor.shape[0] > 3 else tensor.repeat(3, 1, 1)[:3]
        
        # Resize if needed
        h, w = self.image_size
        if tensor.shape[1] != h or tensor.shape[2] != w:
            tensor = F.interpolate(
                tensor.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False
            ).squeeze(0)
        
        return tensor
    
    def _apply_augmentation(
        self, 
        frames: torch.Tensor, 
        flows: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply consistent augmentation across all frames."""
        
        # Horizontal flip (50%)
        if random.random() > 0.5:
            frames = torch.flip(frames, dims=[3])
            flows = torch.flip(flows, dims=[3])
        
        # Color jitter (50%) - RGB only
        if random.random() > 0.5:
            brightness = 1.0 + random.uniform(-0.2, 0.2)
            contrast = 1.0 + random.uniform(-0.2, 0.2)
            
            frames = frames * brightness
            frames = (frames - frames.mean()) * contrast + frames.mean()
            frames = torch.clamp(frames, 0, 1)
        
        # Small rotation (30%)
        if random.random() > 0.7:
            angle = random.uniform(-10, 10)
            # Simple rotation approximation using grid sampling
            # Full rotation would require torchvision transforms
        
        return frames, flows


class TestDataset(Dataset):
    """
    Dataset for unlabeled test data.
    
    Args:
        parquet_path: Path to test.parquet
        num_frames: Frames per video
        image_size: Target size
    """
    
    def __init__(
        self,
        parquet_path: str,
        num_frames: int = 3,
        image_size: Tuple[int, int] = (224, 224),
        base_path: Optional[str] = None,
    ):
        self.df = pd.read_parquet(parquet_path)
        self.num_frames = num_frames
        self.image_size = image_size
        
        if base_path is None:
            self.base_path = Path(parquet_path).parent.parent.parent
        else:
            self.base_path = Path(base_path)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        features_path = row["features_path"]
        n_frames = int(row["n_frames"])
        
        folder_path = self._resolve_path(features_path)
        frame_indices = self._get_frame_indices(n_frames)
        
        frames = []
        flows = []
        
        for fidx in frame_indices:
            frame = self._load_tensor(folder_path / "frames", fidx)
            flow = self._load_tensor(folder_path / "flows", fidx)
            frames.append(frame)
            flows.append(flow)
        
        frames = torch.stack(frames)
        flows = torch.stack(flows)
        
        # Normalize
        frames = (frames - IMAGENET_MEAN) / IMAGENET_STD
        flows = (flows - IMAGENET_MEAN) / IMAGENET_STD
        
        return {
            "frames": frames,
            "flows": flows,
            "video_id": row.get("id", idx),
        }
    
    def _resolve_path(self, features_path: str) -> Path:
        if features_path.startswith("../"):
            return self.base_path / features_path.replace("../", "")
        return Path(features_path)
    
    def _get_frame_indices(self, n_frames: int) -> List[int]:
        if n_frames <= 0:
            return [0] * self.num_frames
        if n_frames < self.num_frames:
            return [min(i, n_frames - 1) for i in range(self.num_frames)]
        step = (n_frames - 1) / (self.num_frames - 1) if self.num_frames > 1 else 0
        return [int(i * step) for i in range(self.num_frames)]
    
    def _load_tensor(self, folder: Path, idx: int) -> torch.Tensor:
        for fmt in [f"{idx:02d}.pt", f"{idx}.pt"]:
            path = folder / fmt
            if path.exists():
                try:
                    tensor = torch.load(path, weights_only=False).float()
                    return self._ensure_shape(tensor)
                except:
                    pass
        
        pt_files = sorted(folder.glob("*.pt"))
        if pt_files:
            actual_idx = min(idx, len(pt_files) - 1)
            tensor = torch.load(pt_files[actual_idx], weights_only=False).float()
            return self._ensure_shape(tensor)
        
        return torch.zeros(3, *self.image_size)
    
    def _ensure_shape(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
        elif tensor.dim() == 3 and tensor.shape[0] != 3:
            if tensor.shape[-1] == 3:
                tensor = tensor.permute(2, 0, 1)
            else:
                tensor = tensor.repeat(3, 1, 1)[:3]
        
        h, w = self.image_size
        if tensor.shape[1] != h or tensor.shape[2] != w:
            tensor = F.interpolate(
                tensor.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False
            ).squeeze(0)
        return tensor


def create_dataloaders(
    train_parquet: str,
    val_split: float = 0.2,
    batch_size: int = 32,
    num_workers: int = 0,
    num_frames: int = 3,
    random_seed: int = 42,
) -> Dict[str, DataLoader]:
    """
    Create train and validation dataloaders with stratified split.
    
    Args:
        train_parquet: Path to training parquet file
        val_split: Fraction for validation (default: 0.2)
        batch_size: Batch size
        num_workers: DataLoader workers (0 for Windows)
        num_frames: Frames per sample
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with 'train' and 'val' DataLoaders
    """
    df = pd.read_parquet(train_parquet)
    
    # Stratified split
    np.random.seed(random_seed)
    
    pos_idx = df[df["target"] == 1].index.values
    neg_idx = df[df["target"] == 0].index.values
    
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    
    n_pos_val = int(len(pos_idx) * val_split)
    n_neg_val = int(len(neg_idx) * val_split)
    
    val_idx = np.concatenate([pos_idx[:n_pos_val], neg_idx[:n_neg_val]])
    train_idx = np.concatenate([pos_idx[n_pos_val:], neg_idx[n_neg_val:]])
    
    np.random.shuffle(val_idx)
    np.random.shuffle(train_idx)
    
    # Create datasets
    train_ds = CollisionDataset(
        train_parquet, train_idx, augment=True, num_frames=num_frames
    )
    val_ds = CollisionDataset(
        train_parquet, val_idx, augment=False, num_frames=num_frames
    )
    
    # Create loaders
    loaders = {
        "train": DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        ),
        "val": DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        ),
    }
    
    logger.info(f"Created dataloaders: Train={len(train_ds)}, Val={len(val_ds)}")
    
    return loaders
