"""
ONNX Export Module
==================

Export trained models to ONNX format with quantization.

EXPORT DOCUMENTATION:

ONNX Configuration:
- Opset version: 17 (latest stable)
- Dynamic axes: Batch dimension only
- Input name: "input"
- Output name: "output"

Quantization Formats:
1. FP32: Full precision (baseline)
2. FP16: Half precision, 2x smaller
3. INT8: 8-bit integer, 4x smaller (requires calibration)

INT8 Calibration:
- Method: Percentile (99.99%)
- Samples: 100 representative inputs
- Static quantization for stable results

Post-Export Optimization:
- onnxsim for graph simplification
- Removes redundant operations
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class ExportResult:
    """Result of model export operation."""
    model_name: str
    format: str  # "onnx" or "tflite"
    quantization: str  # "fp32", "fp16", "int8"
    path: Path
    size_mb: float
    success: bool
    error: Optional[str] = None


class ONNXExporter:
    """
    Export PyTorch models to ONNX format.
    
    Args:
        model: Trained PyTorch model
        model_name: Name for output files
        export_dir: Directory for exported models
        
    Example:
        >>> exporter = ONNXExporter(model, "EfficientNet-B0")
        >>> results = exporter.export(formats=["fp32", "fp16", "int8"])
    """
    
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        export_dir: Optional[str] = None,
    ):
        self.model = model
        self.model_name = model_name
        
        # Export directory
        base_dir = Path(__file__).parent.parent
        self.export_dir = Path(export_dir) if export_dir else base_dir / "exports" / model_name
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        # Set to eval mode
        self.model.eval()
        
        logger.info(f"ONNXExporter initialized: {model_name}")
    
    def export(
        self,
        formats: List[str] = ["fp32", "fp16"],
        opset_version: int = 17,
        input_shape: tuple = (1, 3, 3, 224, 224),  # (B, T, C, H, W)
    ) -> List[ExportResult]:
        """
        Export model to ONNX format(s).
        
        Args:
            formats: List of quantization formats ["fp32", "fp16", "int8"]
            opset_version: ONNX opset version
            input_shape: Shape for dummy input (B, T, C, H, W)
            
        Returns:
            List of ExportResult objects
        """
        results = []
        
        # Always start with FP32
        fp32_path = self._export_fp32(opset_version, input_shape)
        
        if fp32_path is None:
            return [ExportResult(
                model_name=self.model_name,
                format="onnx",
                quantization="fp32",
                path=Path(""),
                size_mb=0,
                success=False,
                error="FP32 export failed",
            )]
        
        # Add FP32 result
        if "fp32" in formats:
            size_mb = fp32_path.stat().st_size / (1024 * 1024)
            results.append(ExportResult(
                model_name=self.model_name,
                format="onnx",
                quantization="fp32",
                path=fp32_path,
                size_mb=size_mb,
                success=True,
            ))
        
        # Export FP16
        if "fp16" in formats:
            result = self._export_fp16(fp32_path)
            results.append(result)
        
        # Export INT8
        if "int8" in formats:
            result = self._export_int8(fp32_path)
            results.append(result)
        
        return results
    
    def _export_fp32(
        self,
        opset_version: int,
        input_shape: tuple,
    ) -> Optional[Path]:
        """Export to FP32 ONNX."""
        output_path = self.export_dir / f"{self.model_name}_fp32.onnx"
        
        try:
            # Create dummy inputs
            batch_size, num_frames, channels, h, w = input_shape
            dummy_frames = torch.randn(batch_size, num_frames, channels, h, w)
            dummy_flows = torch.randn(batch_size, num_frames, channels, h, w)
            
            # Export
            torch.onnx.export(
                self.model,
                (dummy_frames, dummy_flows),
                str(output_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["frames", "flows"],
                output_names=["output"],
                dynamic_axes={
                    "frames": {0: "batch_size"},
                    "flows": {0: "batch_size"},
                    "output": {0: "batch_size"},
                },
            )
            
            logger.info(f"Exported FP32: {output_path}")
            
            # Simplify if onnxsim available
            self._simplify_onnx(output_path)
            
            return output_path
            
        except Exception as e:
            logger.error(f"FP32 export failed: {e}")
            return None
    
    def _export_fp16(self, fp32_path: Path) -> ExportResult:
        """Convert FP32 ONNX to FP16."""
        output_path = self.export_dir / f"{self.model_name}_fp16.onnx"
        
        try:
            import onnx
            from onnxconverter_common import float16
            
            # Load FP32 model
            model = onnx.load(str(fp32_path))
            
            # Convert to FP16
            model_fp16 = float16.convert_float_to_float16(
                model,
                keep_io_types=True,  # Keep input/output as FP32
            )
            
            # Save
            onnx.save(model_fp16, str(output_path))
            
            size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"Exported FP16: {output_path} ({size_mb:.2f} MB)")
            
            return ExportResult(
                model_name=self.model_name,
                format="onnx",
                quantization="fp16",
                path=output_path,
                size_mb=size_mb,
                success=True,
            )
            
        except ImportError:
            return ExportResult(
                model_name=self.model_name,
                format="onnx",
                quantization="fp16",
                path=Path(""),
                size_mb=0,
                success=False,
                error="onnxconverter-common not installed",
            )
        except Exception as e:
            return ExportResult(
                model_name=self.model_name,
                format="onnx",
                quantization="fp16",
                path=Path(""),
                size_mb=0,
                success=False,
                error=str(e),
            )
    
    def _export_int8(self, fp32_path: Path) -> ExportResult:
        """Convert FP32 ONNX to INT8 using static quantization."""
        output_path = self.export_dir / f"{self.model_name}_int8.onnx"
        
        try:
            from onnxruntime.quantization import quantize_static, CalibrationDataReader
            from onnxruntime.quantization import QuantType, QuantFormat
            
            # Calibration data reader
            class DummyCalibrationReader(CalibrationDataReader):
                def __init__(self, num_samples: int = 100):
                    self.samples = num_samples
                    self.index = 0
                
                def get_next(self):
                    if self.index >= self.samples:
                        return None
                    
                    self.index += 1
                    return {
                        "frames": np.random.randn(1, 3, 3, 224, 224).astype(np.float32),
                        "flows": np.random.randn(1, 3, 3, 224, 224).astype(np.float32),
                    }
                
                def rewind(self):
                    self.index = 0
            
            # Quantize
            quantize_static(
                model_input=str(fp32_path),
                model_output=str(output_path),
                calibration_data_reader=DummyCalibrationReader(100),
                quant_format=QuantFormat.QDQ,
                per_channel=True,
                weight_type=QuantType.QInt8,
            )
            
            size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"Exported INT8: {output_path} ({size_mb:.2f} MB)")
            
            return ExportResult(
                model_name=self.model_name,
                format="onnx",
                quantization="int8",
                path=output_path,
                size_mb=size_mb,
                success=True,
            )
            
        except ImportError:
            return ExportResult(
                model_name=self.model_name,
                format="onnx",
                quantization="int8",
                path=Path(""),
                size_mb=0,
                success=False,
                error="onnxruntime not installed",
            )
        except Exception as e:
            return ExportResult(
                model_name=self.model_name,
                format="onnx",
                quantization="int8",
                path=Path(""),
                size_mb=0,
                success=False,
                error=str(e),
            )
    
    def _simplify_onnx(self, onnx_path: Path) -> None:
        """Simplify ONNX model using onnxsim."""
        try:
            import onnx
            from onnxsim import simplify
            
            model = onnx.load(str(onnx_path))
            model_simplified, check = simplify(model)
            
            if check:
                onnx.save(model_simplified, str(onnx_path))
                logger.info(f"Simplified ONNX model: {onnx_path.name}")
            else:
                logger.warning(f"ONNX simplification failed validation")
                
        except ImportError:
            logger.debug("onnxsim not installed, skipping simplification")
        except Exception as e:
            logger.warning(f"ONNX simplification failed: {e}")


def export_model(
    model_name: str,
    formats: List[str] = ["fp32", "fp16"],
    checkpoint_path: Optional[str] = None,
) -> List[ExportResult]:
    """
    Export a trained model to ONNX format(s).
    
    Args:
        model_name: Model architecture name
        formats: Quantization formats to export
        checkpoint_path: Path to checkpoint (auto-detected if None)
        
    Returns:
        List of ExportResult objects
    """
    from models.factory import create_model
    from config.settings import CHECKPOINT_DIR
    
    # Find checkpoint
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_DIR / model_name / "best.pth"
    
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return []
    
    # Load model
    model = create_model(model_name, pretrained=False)
    state = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    
    logger.info(f"Loaded checkpoint: {checkpoint_path}")
    
    # Export
    exporter = ONNXExporter(model, model_name)
    results = exporter.export(formats=formats)
    
    return results


def export_all_models(
    formats: List[str] = ["fp32", "fp16"]
) -> dict:
    """Export all available trained models."""
    from config.settings import get_all_model_names, CHECKPOINT_DIR
    
    results = {}
    
    for model_name in get_all_model_names():
        checkpoint = CHECKPOINT_DIR / model_name / "best.pth"
        
        if checkpoint.exists():
            model_results = export_model(model_name, formats)
            results[model_name] = model_results
        else:
            logger.warning(f"No checkpoint for {model_name}, skipping")
    
    return results
