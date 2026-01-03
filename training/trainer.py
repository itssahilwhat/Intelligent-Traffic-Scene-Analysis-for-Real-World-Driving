"""
Training Pipeline Module
========================

Complete training loop with logging, checkpointing, and metrics.

TRAINING DOCUMENTATION:

Optimizer: AdamW
- Learning rate: 1e-4 (default)
- Betas: (0.9, 0.999)
- Weight decay: 1e-4
- Epsilon: 1e-8

Scheduler: ReduceLROnPlateau
- Monitor: validation loss
- Factor: 0.5 (halve LR on plateau)
- Patience: 5 epochs
- Min LR: 1e-7

Early Stopping:
- Patience: 10 epochs
- Monitor: validation loss

Checkpointing:
- Save best model (by validation loss)
- Save final model
- Save training history

Mixed Precision:
- Enabled by default (torch.cuda.amp)
- Improves speed and memory usage
"""

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .losses import get_loss_function
from .metrics import compute_metrics, format_metrics

logger = logging.getLogger(__name__)


@dataclass
class TrainingHistory:
    """Container for training history."""
    train_loss: List[float]
    val_loss: List[float]
    train_metrics: List[Dict]
    val_metrics: List[Dict]
    learning_rates: List[float]
    epoch_times: List[float]
    best_epoch: int
    best_val_loss: float


class Trainer:
    """
    Complete training pipeline for collision detection.
    
    Args:
        model: PyTorch model
        model_name: Name for saving checkpoints
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        lr: Learning rate
        weight_decay: L2 regularization
        device: 'cuda', 'cpu', or 'auto'
        use_amp: Enable mixed precision training
        loss_type: 'bce' or 'focal'
        checkpoint_dir: Directory for saving checkpoints
        results_dir: Directory for saving results
        
    Example:
        >>> trainer = Trainer(model, "EfficientNet-B0", train_loader, val_loader)
        >>> history = trainer.train(epochs=50)
    """
    
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        device: Optional[str] = None,
        use_amp: bool = True,
        loss_type: str = "bce",
        checkpoint_dir: Optional[str] = None,
        results_dir: Optional[str] = None,
    ):
        self.model = model
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.use_amp = use_amp and torch.cuda.is_available()
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Directories
        base_dir = Path(__file__).parent.parent
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else base_dir / "checkpoints" / model_name
        self.results_dir = Path(results_dir) if results_dir else base_dir / "results" / model_name
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function
        self.criterion = get_loss_function(loss_type)
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
            eps=1e-8,
        )
        
        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=True,
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training state
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.patience_counter = 0
        
        # Logging
        self.log_file = self.results_dir / "training.log"
        
        logger.info(f"Trainer initialized: device={self.device}, amp={self.use_amp}")
    
    def train(
        self,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        gradient_clip: float = 1.0,
    ) -> TrainingHistory:
        """
        Train the model.
        
        Args:
            epochs: Maximum training epochs
            early_stopping_patience: Stop if no improvement for N epochs
            gradient_clip: Gradient clipping norm
            
        Returns:
            TrainingHistory object with all training data
        """
        self._log(f"Starting training: {self.model_name}")
        self._log(f"Epochs: {epochs}, Device: {self.device}")
        self._log(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
        
        # Save config
        self._save_config(epochs, early_stopping_patience, gradient_clip)
        
        # History tracking
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": [],
            "learning_rates": [],
            "epoch_times": [],
        }
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training phase
            train_loss, train_preds, train_labels = self._run_epoch(
                self.train_loader, training=True, gradient_clip=gradient_clip
            )
            train_metrics = compute_metrics(train_labels, train_preds)
            
            # Validation phase
            val_loss, val_preds, val_labels = self._run_epoch(
                self.val_loader, training=False
            )
            val_metrics = compute_metrics(val_labels, val_preds)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            # Track history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_metrics"].append(train_metrics)
            history["val_metrics"].append(val_metrics)
            history["learning_rates"].append(current_lr)
            history["epoch_times"].append(time.time() - epoch_start)
            
            # Log progress
            self._log(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f} | "
                f"LR: {current_lr:.2e}"
            )
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0
                self._save_checkpoint("best.pth", epoch, val_loss, val_metrics)
                self._log(f"  → New best model saved!")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                self._log(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Save final model
        self._save_checkpoint("final.pth", epoch, val_loss, val_metrics)
        
        # Save training history
        self._save_history(history)
        
        total_time = time.time() - start_time
        self._log(f"Training complete! Total time: {total_time/60:.1f} min")
        self._log(f"Best epoch: {self.best_epoch+1}, Best val loss: {self.best_val_loss:.4f}")
        
        return TrainingHistory(
            train_loss=history["train_loss"],
            val_loss=history["val_loss"],
            train_metrics=history["train_metrics"],
            val_metrics=history["val_metrics"],
            learning_rates=history["learning_rates"],
            epoch_times=history["epoch_times"],
            best_epoch=self.best_epoch,
            best_val_loss=self.best_val_loss,
        )
    
    def _run_epoch(
        self,
        loader: DataLoader,
        training: bool = True,
        gradient_clip: float = 1.0,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Run a single epoch."""
        if training:
            self.model.train()
        else:
            self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        context = torch.enable_grad() if training else torch.no_grad()
        
        with context:
            for batch in loader:
                frames = batch["frames"].to(self.device)
                flows = batch["flows"].to(self.device)
                labels = batch["label"].to(self.device).unsqueeze(1)
                
                if training:
                    self.optimizer.zero_grad()
                
                # Forward pass
                if self.use_amp and training:
                    with autocast():
                        outputs = self.model(frames, flows)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(frames, flows)
                    loss = self.criterion(outputs, labels)
                
                if training:
                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                        self.optimizer.step()
                
                # Track metrics
                total_loss += loss.item() * len(labels)
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_preds.extend(probs.flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
        
        avg_loss = total_loss / len(loader.dataset)
        return avg_loss, np.array(all_preds), np.array(all_labels)
    
    def _save_checkpoint(
        self,
        filename: str,
        epoch: int,
        val_loss: float,
        val_metrics: Dict,
    ):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "val_metrics": val_metrics,
            "model_name": self.model_name,
        }
        
        torch.save(checkpoint, self.checkpoint_dir / filename)
    
    def _save_config(self, epochs: int, patience: int, grad_clip: float):
        """Save training configuration."""
        config = {
            "model_name": self.model_name,
            "epochs": epochs,
            "early_stopping_patience": patience,
            "gradient_clip": grad_clip,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "weight_decay": self.optimizer.param_groups[0]["weight_decay"],
            "device": str(self.device),
            "use_amp": self.use_amp,
            "train_samples": len(self.train_loader.dataset),
            "val_samples": len(self.val_loader.dataset),
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(self.results_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
    
    def _save_history(self, history: Dict):
        """Save training history."""
        # Save as JSON
        with open(self.results_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2, default=float)
        
        # Save metrics as CSV
        rows = []
        for i, (train_m, val_m) in enumerate(zip(
            history["train_metrics"], history["val_metrics"]
        )):
            row = {
                "epoch": i + 1,
                "train_loss": history["train_loss"][i],
                "val_loss": history["val_loss"][i],
                "lr": history["learning_rates"][i],
            }
            for k, v in train_m.items():
                row[f"train_{k}"] = v
            for k, v in val_m.items():
                row[f"val_{k}"] = v
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(self.results_dir / "metrics.csv", index=False)
    
    def _log(self, message: str):
        """Log message to file and console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        print(log_message)
        logger.info(message)
        
        with open(self.log_file, "a") as f:
            f.write(log_message + "\n")


def train_model(
    model_name: str,
    epochs: int = 50,
    batch_size: int = 32,
    loss_type: str = "bce",
    train_parquet: Optional[str] = None,
) -> TrainingHistory:
    """
    Convenience function to train a model by name.
    
    Args:
        model_name: Name of model to train
        epochs: Training epochs
        batch_size: Batch size
        loss_type: 'bce' or 'focal'
        train_parquet: Path to training data (auto-detected if None)
        
    Returns:
        TrainingHistory object
    """
    from models.factory import create_model
    from data.dataset import create_dataloaders
    from config.settings import TRAIN_PARQUET, get_model_config
    
    # Get model config
    config = get_model_config(model_name)
    
    # Create model
    model = create_model(
        model_name,
        pretrained=True,
        dropout=config.dropout,
    )
    
    # Create dataloaders
    parquet_path = train_parquet or str(TRAIN_PARQUET)
    loaders = create_dataloaders(
        parquet_path,
        batch_size=batch_size,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        model_name=model_name,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        loss_type=loss_type,
    )
    
    # Train
    history = trainer.train(epochs=epochs)
    
    return history
