"""
Inference Module
================

Model inference for collision prediction.

INFERENCE DOCUMENTATION:

Input Processing:
1. Load 6-channel tensor or assemble from components
2. Normalize RGB with ImageNet stats
3. Expand batch dimension: (6, H, W) → (1, 6, H, W)

Forward Pass:
1. Model outputs raw logits
2. Apply sigmoid for probabilities
3. Compare to threshold for binary prediction

Thresholding:
- Default: 0.5 (probability > 0.5 → crash predicted)
- Can be tuned for precision/recall tradeoff
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Predictor:
    """
    Inference engine for collision prediction.
    
    Args:
        model: Trained PyTorch model
        device: 'cuda', 'cpu', or 'auto'
        threshold: Decision threshold (default: 0.5)
        
    Example:
        >>> predictor = Predictor(model)
        >>> prob = predictor.predict(frames, flows)
        >>> is_crash = prob > 0.5
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        threshold: float = 0.5,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = model.to(self.device)
        self.model.eval()
        self.threshold = threshold
        
        logger.info(f"Predictor initialized on {self.device}")
    
    @torch.no_grad()
    def predict(
        self,
        frames: Union[torch.Tensor, np.ndarray],
        flows: Union[torch.Tensor, np.ndarray],
    ) -> float:
        """
        Predict collision probability for single sample.
        
        Args:
            frames: RGB frames (T, 3, H, W) or (3, H, W)
            flows: Optical flow (T, 3, H, W) or (3, H, W)
            
        Returns:
            Collision probability in [0, 1]
        """
        # Convert to tensors
        if isinstance(frames, np.ndarray):
            frames = torch.from_numpy(frames).float()
        if isinstance(flows, np.ndarray):
            flows = torch.from_numpy(flows).float()
        
        # Add batch dimension
        if frames.dim() == 3:
            frames = frames.unsqueeze(0).unsqueeze(0)
            flows = flows.unsqueeze(0).unsqueeze(0)
        elif frames.dim() == 4:
            frames = frames.unsqueeze(0)
            flows = flows.unsqueeze(0)
        
        frames = frames.to(self.device)
        flows = flows.to(self.device)
        
        # Forward pass
        logits = self.model(frames, flows)
        prob = torch.sigmoid(logits).item()
        
        return prob
    
    @torch.no_grad()
    def predict_batch(
        self,
        frames: Union[torch.Tensor, np.ndarray],
        flows: Union[torch.Tensor, np.ndarray],
    ) -> np.ndarray:
        """
        Predict collision probabilities for batch.
        
        Args:
            frames: Batch of frames (B, T, 3, H, W)
            flows: Batch of flows (B, T, 3, H, W)
            
        Returns:
            Array of probabilities (B,)
        """
        if isinstance(frames, np.ndarray):
            frames = torch.from_numpy(frames).float()
        if isinstance(flows, np.ndarray):
            flows = torch.from_numpy(flows).float()
        
        frames = frames.to(self.device)
        flows = flows.to(self.device)
        
        logits = self.model(frames, flows)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        
        return probs
    
    def classify(
        self,
        frames: Union[torch.Tensor, np.ndarray],
        flows: Union[torch.Tensor, np.ndarray],
    ) -> Tuple[bool, float]:
        """
        Classify as crash/safe with confidence.
        
        Args:
            frames: Input frames
            flows: Input flows
            
        Returns:
            Tuple of (is_crash, probability)
        """
        prob = self.predict(frames, flows)
        is_crash = prob >= self.threshold
        return is_crash, prob
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader,
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Evaluate model on dataloader.
        
        Args:
            dataloader: PyTorch DataLoader with labeled data
            
        Returns:
            Dictionary with predictions, labels, and metrics
        """
        from training.metrics import compute_metrics
        
        all_probs = []
        all_labels = []
        
        for batch in dataloader:
            frames = batch["frames"].to(self.device)
            flows = batch["flows"].to(self.device)
            labels = batch["label"].numpy()
            
            logits = self.model(frames, flows)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            
            all_probs.extend(probs)
            all_labels.extend(labels)
        
        probs = np.array(all_probs)
        labels = np.array(all_labels)
        
        metrics = compute_metrics(labels, probs, self.threshold)
        
        return {
            "predictions": probs,
            "labels": labels,
            "metrics": metrics,
        }


def load_predictor(
    model_name: str,
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None,
) -> Predictor:
    """
    Load a trained model for inference.
    
    Args:
        model_name: Name of model architecture
        checkpoint_path: Path to checkpoint (auto-detected if None)
        device: Device to use
        
    Returns:
        Predictor instance
    """
    from models.factory import create_model
    from config.settings import CHECKPOINT_DIR
    
    # Find checkpoint
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_DIR / model_name / "best.pth"
    
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load model
    model = create_model(model_name, pretrained=False)
    
    # Load weights
    state = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    
    logger.info(f"Loaded checkpoint: {checkpoint_path}")
    
    return Predictor(model, device=device)


def evaluate_model(
    model_name: str,
    test_loader,
    checkpoint_path: Optional[str] = None,
) -> Dict:
    """
    Evaluate a model on test data.
    
    Args:
        model_name: Model architecture name
        test_loader: DataLoader for test data
        checkpoint_path: Path to checkpoint
        
    Returns:
        Evaluation results
    """
    predictor = load_predictor(model_name, checkpoint_path)
    results = predictor.evaluate(test_loader)
    
    # Add calibration analysis
    probs = results["predictions"]
    results["calibration"] = {
        "mean_prob": float(np.mean(probs)),
        "std_prob": float(np.std(probs)),
        "positive_rate_0.5": float(np.mean(probs >= 0.5)),
        "positive_rate_0.3": float(np.mean(probs >= 0.3)),
        "positive_rate_0.7": float(np.mean(probs >= 0.7)),
    }
    
    return results
