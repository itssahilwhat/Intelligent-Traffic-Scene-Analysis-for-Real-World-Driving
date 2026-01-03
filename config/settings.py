"""
Centralized Configuration for Dashcam Collision Prediction
===========================================================

All hyperparameters, paths, and model configurations in one place.
Every value is documented for reproducibility.

REPRODUCIBILITY DOCUMENTATION:
- Random seed: 42 (used for train/val splits, weight init)
- All hyperparameters match IEEE T-ITS paper specifications
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).parent.parent  # MP-02-Codes/
PROJECT_ROOT = BASE_DIR.parent  # nexar-collision-prediction-main/

# Data paths
DATA_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_PARQUET = DATA_DIR / "train.parquet"
TEST_PARQUET = DATA_DIR / "test.parquet"

# Output paths
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
EXPORT_DIR = BASE_DIR / "exports"
RESULTS_DIR = BASE_DIR / "results"


# =============================================================================
# ENUMS
# =============================================================================

class ModelArch(str, Enum):
    """
    Supported model architectures.
    
    Each model was selected for specific deployment characteristics:
    - FastViT: Hybrid ViT, best accuracy/speed tradeoff
    - EfficientNet: Compound-scaled CNN, proven baseline
    - MobileNetV3: Ultra-lightweight for edge devices
    - ConvNeXt: Modern pure CNN architecture
    - ResNet: Classic baseline for comparison
    """
    FASTVIT_T8 = "fastvit_t8"
    FASTVIT_T12 = "fastvit_t12"
    EFFICIENTNET_B0 = "efficientnet_b0"
    MOBILENETV3_SMALL = "mobilenetv3_small"
    CONVNEXT_TINY = "convnext_tiny"
    RESNET18 = "resnet18"


class QuantizationFormat(str, Enum):
    """Export quantization formats."""
    FP32 = "fp32"   # Full precision (baseline)
    FP16 = "fp16"   # Half precision (2x smaller)
    INT8 = "int8"   # 8-bit integer (4x smaller)


# =============================================================================
# PREPROCESSING CONFIGURATION
# =============================================================================

@dataclass
class VideoConfig:
    """
    Video frame extraction parameters.
    
    DOCUMENTATION:
    - Library: OpenCV 4.x (cv2.VideoCapture)
    - Frame sampling: Extract every Nth frame
    - For 30 FPS input with sample_rate=15 → 2 FPS output
    """
    source_fps: int = 30           # Expected input video FPS
    sample_rate: int = 15          # Extract every N frames (30/15 = 2 FPS)
    target_resolution: Tuple[int, int] = (1280, 720)  # Expected resolution
    codec: str = "mp4v"            # Expected codec


@dataclass
class OpticalFlowConfig:
    """
    Farneback Optical Flow parameters.
    
    DOCUMENTATION:
    - Algorithm: Farneback (chosen over Lucas-Kanade for dense flow)
    - Implementation: cv2.calcOpticalFlowFarneback()
    - Input: Consecutive grayscale frames (0-255 uint8)
    - Output: Two-channel array [Vx, Vy] with shape (H, W, 2)
    
    WHY FARNEBACK?
    - Provides dense flow (every pixel) vs sparse (Lucas-Kanade)
    - Better captures vehicle motion patterns
    - Reasonable computational cost
    """
    pyr_scale: float = 0.5      # Pyramid scale factor
    levels: int = 3             # Number of pyramid levels
    winsize: int = 15           # Averaging window size
    iterations: int = 3         # Iterations at each pyramid level
    poly_n: int = 5             # Polynomial expansion neighborhood size
    poly_sigma: float = 1.2     # Standard deviation for Gaussian
    flags: int = 0              # 0 = no flags, can use cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    
    # Flow normalization
    max_flow: float = 20.0      # Clip flow values to [-max_flow, max_flow]
    normalize_output: bool = True  # Scale to [0, 1] for visualization


@dataclass
class VehicleDetectionConfig:
    """
    YOLOv8 vehicle detection configuration.
    
    DOCUMENTATION:
    - Model: yolov8n-seg.pt (nano segmentation, 3.4M params)
    - Classes: COCO vehicle classes
      - 2: car
      - 5: bus  
      - 7: truck
    - Output: Binary mask where 1=vehicle, 0=background
    """
    model_name: str = "yolov8n-seg.pt"
    confidence_threshold: float = 0.5
    vehicle_classes: Tuple[int, ...] = (2, 5, 7)  # car, bus, truck
    
    # Mask morphology
    apply_morphology: bool = True
    kernel_size: int = 5
    close_iterations: int = 2  # cv2.MORPH_CLOSE iterations
    
    # Multi-vehicle handling
    combine_masks: str = "union"  # "union" (OR all masks) or "largest"


# =============================================================================
# IMAGE CONFIGURATION
# =============================================================================

@dataclass  
class ImageConfig:
    """
    Image preprocessing configuration.
    
    DOCUMENTATION:
    - Resize: 224x224 using cv2.INTER_LINEAR (bilinear interpolation)
    - Normalization: ImageNet statistics (applied AFTER resize)
    - Channel order: [R, G, B, Vx_masked, Vy_masked, Mask]
    """
    size: Tuple[int, int] = (224, 224)
    interpolation: str = "bilinear"  # cv2.INTER_LINEAR
    
    # ImageNet normalization (for RGB channels only)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # Input channels
    num_channels: int = 6  # RGB(3) + Flow(2) + Mask(1)
    
    # Channel order documentation
    # Channel 0: Red (normalized with ImageNet mean/std)
    # Channel 1: Green (normalized with ImageNet mean/std)
    # Channel 2: Blue (normalized with ImageNet mean/std)
    # Channel 3: Optical flow Vx (masked, scaled to [0,1])
    # Channel 4: Optical flow Vy (masked, scaled to [0,1])
    # Channel 5: Vehicle mask (binary 0/1)


# =============================================================================
# DATA AUGMENTATION
# =============================================================================

@dataclass
class AugmentationConfig:
    """
    Training data augmentation parameters.
    
    DOCUMENTATION:
    - Applied ONLY during training, not validation/test
    - Augmentations applied consistently across all frames in sequence
    """
    # Geometric
    horizontal_flip_prob: float = 0.5
    rotation_range: Tuple[float, float] = (-10.0, 10.0)  # degrees
    rotation_prob: float = 0.3
    
    # Photometric (RGB channels only)
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    saturation_range: Tuple[float, float] = (0.8, 1.2)
    color_jitter_prob: float = 0.5
    
    # Noise
    gaussian_noise_std: float = 0.02
    gaussian_noise_prob: float = 0.1
    
    # Blur
    gaussian_blur_kernels: Tuple[int, ...] = (3, 5)
    gaussian_blur_prob: float = 0.1
    
    # Not used (documented for completeness)
    mixup_alpha: float = 0.0        # Disabled
    cutmix_alpha: float = 0.0       # Disabled
    test_time_augmentation: bool = False


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """
    Training hyperparameters.
    
    DOCUMENTATION:
    - Optimizer: AdamW (better weight decay handling than Adam)
    - Scheduler: ReduceLROnPlateau (monitors validation loss)
    - Early stopping: Halts if val loss doesn't improve for N epochs
    """
    # Core training
    epochs: int = 50
    batch_size: int = 32
    num_workers: int = 0  # 0 for Windows compatibility
    
    # Optimizer: AdamW
    learning_rate: float = 1e-4
    beta1: float = 0.9              # Momentum
    beta2: float = 0.999            # RMSprop
    weight_decay: float = 1e-4      # L2 regularization
    eps: float = 1e-8               # Numerical stability
    
    # Scheduler: ReduceLROnPlateau
    scheduler_factor: float = 0.5   # Multiply LR by this on plateau
    scheduler_patience: int = 5     # Epochs without improvement
    scheduler_min_lr: float = 1e-7  # Minimum learning rate
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_monitor: str = "val_loss"  # or "val_accuracy"
    
    # Gradient clipping
    gradient_clip_norm: float = 1.0
    
    # Mixed precision
    use_amp: bool = True  # Automatic Mixed Precision
    
    # Reproducibility
    random_seed: int = 42
    deterministic: bool = True
    
    # Data split
    val_split: float = 0.2  # 80% train, 20% validation
    stratified: bool = True  # Maintain class balance in splits
    
    # Multi-frame
    num_frames: int = 3  # Frames per video sample


@dataclass
class LossConfig:
    """
    Loss function configuration.
    
    DOCUMENTATION:
    - Default: BCEWithLogitsLoss (numerically stable)
    - Alternative: FocalLoss for class imbalance
    - Model outputs raw logits, sigmoid applied in loss
    """
    loss_type: str = "bce"  # "bce" or "focal"
    
    # BCEWithLogitsLoss
    pos_weight: float = 1.0  # Weight for positive class (adjust for imbalance)
    
    # FocalLoss parameters
    focal_alpha: float = 0.25  # Balance factor
    focal_gamma: float = 2.0   # Focusing parameter
    
    # Label smoothing (only for V2 models)
    label_smoothing: float = 0.0  # 0.0 = disabled, 0.1 = typical


# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    architecture: ModelArch
    pretrained: bool = True
    dropout: float = 0.35
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    
    # Feature dimensions (filled by factory)
    feature_dim: int = 0
    param_count: str = ""
    
    @property
    def checkpoint_dir(self) -> Path:
        return CHECKPOINT_DIR / self.name
    
    @property
    def best_checkpoint(self) -> Path:
        return self.checkpoint_dir / "best.pth"


# Model registry with exact specifications
MODELS: Dict[str, ModelConfig] = {
    "FastViT-T8": ModelConfig(
        name="FastViT-T8",
        architecture=ModelArch.FASTVIT_T8,
        dropout=0.35,
        learning_rate=1e-4,
        feature_dim=384,
        param_count="3.26M",
    ),
    "FastViT-T12": ModelConfig(
        name="FastViT-T12",
        architecture=ModelArch.FASTVIT_T12,
        dropout=0.35,
        learning_rate=1e-4,
        feature_dim=384,
        param_count="6.8M",
    ),
    "EfficientNet-B0": ModelConfig(
        name="EfficientNet-B0",
        architecture=ModelArch.EFFICIENTNET_B0,
        dropout=0.35,
        learning_rate=1e-4,
        feature_dim=1280,
        param_count="4.01M",
    ),
    "MobileNetV3-Small": ModelConfig(
        name="MobileNetV3-Small",
        architecture=ModelArch.MOBILENETV3_SMALL,
        dropout=0.30,
        learning_rate=1e-4,
        feature_dim=576,
        param_count="0.93M",
    ),
    "ConvNeXt-Tiny": ModelConfig(
        name="ConvNeXt-Tiny",
        architecture=ModelArch.CONVNEXT_TINY,
        dropout=0.40,
        learning_rate=5e-5,
        weight_decay=0.05,
        feature_dim=768,
        param_count="28.6M",
    ),
    "ResNet-18": ModelConfig(
        name="ResNet-18",
        architecture=ModelArch.RESNET18,
        dropout=0.35,
        learning_rate=1e-4,
        feature_dim=512,
        param_count="11.19M",
    ),
}


# =============================================================================
# EXPORT CONFIGURATION
# =============================================================================

@dataclass
class ExportConfig:
    """
    Model export configuration.
    
    DOCUMENTATION:
    - ONNX opset: 17 (latest stable)
    - INT8 calibration: 100 representative samples
    - Input shape: (1, 6, 224, 224) for single frame
    """
    onnx_opset: int = 17
    simplify_onnx: bool = True  # Use onnxsim
    
    # INT8 quantization
    calibration_samples: int = 100
    calibration_method: str = "percentile"  # or "entropy"
    calibration_percentile: float = 99.99
    
    # Input/output naming
    input_name: str = "input"
    output_name: str = "output"


# =============================================================================
# METRICS CONFIGURATION
# =============================================================================

@dataclass
class MetricsConfig:
    """
    Evaluation metrics configuration.
    
    DOCUMENTATION:
    - Decision threshold: 0.5 (probability > 0.5 → crash predicted)
    - All metrics computed at this threshold unless specified
    """
    decision_threshold: float = 0.5
    
    # Metrics to compute
    compute_accuracy: bool = True
    compute_precision: bool = True
    compute_recall: bool = True
    compute_f1: bool = True
    compute_specificity: bool = True
    compute_auroc: bool = True
    compute_auprc: bool = True  # Average Precision
    compute_mcc: bool = True    # Matthews Correlation Coefficient


# =============================================================================
# DEFAULT CONFIG INSTANCES
# =============================================================================

VIDEO_CONFIG = VideoConfig()
OPTICAL_FLOW_CONFIG = OpticalFlowConfig()
VEHICLE_DETECTION_CONFIG = VehicleDetectionConfig()
IMAGE_CONFIG = ImageConfig()
AUGMENTATION_CONFIG = AugmentationConfig()
TRAINING_CONFIG = TrainingConfig()
LOSS_CONFIG = LossConfig()
EXPORT_CONFIG = ExportConfig()
METRICS_CONFIG = MetricsConfig()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_model_config(name: str) -> ModelConfig:
    """Get model configuration by name."""
    if name not in MODELS:
        available = list(MODELS.keys())
        raise ValueError(f"Unknown model: {name}. Available: {available}")
    return MODELS[name]


def get_all_model_names() -> List[str]:
    """Get list of all available model names."""
    return list(MODELS.keys())


def ensure_dirs() -> None:
    """Create all required directories."""
    for d in [CHECKPOINT_DIR, EXPORT_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    (EXPORT_DIR / "onnx").mkdir(exist_ok=True)
    (EXPORT_DIR / "tflite").mkdir(exist_ok=True)


# =============================================================================
# PRINT CONFIG SUMMARY
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DASHCAM COLLISION PREDICTION - CONFIGURATION SUMMARY")
    print("=" * 70)
    
    print("\n[PATHS]")
    print(f"  Base Dir:     {BASE_DIR}")
    print(f"  Data Dir:     {DATA_DIR}")
    print(f"  Checkpoints:  {CHECKPOINT_DIR}")
    
    print("\n[PREPROCESSING]")
    print(f"  Optical Flow: Farneback (pyr_scale={OPTICAL_FLOW_CONFIG.pyr_scale}, "
          f"levels={OPTICAL_FLOW_CONFIG.levels}, winsize={OPTICAL_FLOW_CONFIG.winsize})")
    print(f"  Vehicle Det:  {VEHICLE_DETECTION_CONFIG.model_name} "
          f"(conf={VEHICLE_DETECTION_CONFIG.confidence_threshold})")
    
    print("\n[IMAGE]")
    print(f"  Size:         {IMAGE_CONFIG.size}")
    print(f"  Channels:     {IMAGE_CONFIG.num_channels} (RGB + Flow + Mask)")
    print(f"  Normalize:    ImageNet (mean={IMAGE_CONFIG.mean})")
    
    print("\n[TRAINING]")
    print(f"  Epochs:       {TRAINING_CONFIG.epochs}")
    print(f"  Batch Size:   {TRAINING_CONFIG.batch_size}")
    print(f"  LR:           {TRAINING_CONFIG.learning_rate}")
    print(f"  Optimizer:    AdamW (wd={TRAINING_CONFIG.weight_decay})")
    print(f"  Seed:         {TRAINING_CONFIG.random_seed}")
    
    print("\n[MODELS]")
    for name, cfg in MODELS.items():
        print(f"  - {name}: {cfg.architecture.value}, {cfg.param_count}, "
              f"lr={cfg.learning_rate}")
    
    print("\n" + "=" * 70)
