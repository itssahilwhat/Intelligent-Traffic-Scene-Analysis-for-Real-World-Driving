#!/usr/bin/env python
"""
Dashcam Collision Prediction
============================

Main entry point for training, exporting, and evaluating collision detection models.

Usage:
    python main.py train --model EfficientNet-B0 --epochs 50
    python main.py export --model EfficientNet-B0 --format fp32 fp16 int8
    python main.py evaluate --model EfficientNet-B0
    python main.py list

Models Available:
    - FastViT-T8, FastViT-T12 (Hybrid Vision Transformers)
    - EfficientNet-B0 (Compound-scaled CNN)
    - MobileNetV3-Small (Lightweight Mobile CNN)
    - ConvNeXt-Tiny (Modern CNN)
    - ResNet-18 (Classic Baseline)
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_train(args):
    """Train a model."""
    from training.trainer import train_model
    from config.settings import get_all_model_names
    
    models = get_all_model_names() if args.model.lower() == "all" else [args.model]
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"TRAINING: {model_name}")
        print(f"{'='*60}\n")
        
        train_model(
            model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            loss_type=args.loss,
        )


def cmd_export(args):
    """Export model to ONNX."""
    from export.onnx_exporter import export_model, export_all_models
    
    if args.model.lower() == "all":
        results = export_all_models(formats=args.format)
        for model_name, model_results in results.items():
            for r in model_results:
                status = "✓" if r.success else "✗"
                print(f"{status} {model_name} {r.quantization}: {r.size_mb:.2f} MB")
    else:
        results = export_model(args.model, formats=args.format)
        for r in results:
            status = "✓" if r.success else "✗"
            print(f"{status} {r.model_name} {r.quantization}: {r.size_mb:.2f} MB")


def cmd_evaluate(args):
    """Evaluate model on test data."""
    from inference.predictor import evaluate_model
    from data.dataset import TestDataset
    from config.settings import TEST_PARQUET, get_all_model_names
    from torch.utils.data import DataLoader
    
    models = get_all_model_names() if args.model.lower() == "all" else [args.model]
    
    test_ds = TestDataset(str(TEST_PARQUET))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"EVALUATING: {model_name}")
        print(f"{'='*60}")
        
        try:
            results = evaluate_model(model_name, test_loader)
            cal = results["calibration"]
            print(f"Mean Probability: {cal['mean_prob']:.4f}")
            print(f"Positive Rate (0.5): {cal['positive_rate_0.5']*100:.1f}%")
            print(f"Std Probability: {cal['std_prob']:.4f}")
        except Exception as e:
            print(f"Error: {e}")


def cmd_list(args):
    """List available models."""
    from models.factory import list_models
    list_models()


def cmd_preprocess(args):
    """Run preprocessing pipeline on a video."""
    from preprocessing.pipeline import PreprocessingPipeline
    
    pipeline = PreprocessingPipeline(
        frame_sample_rate=args.sample_rate,
        use_yolo=not args.no_yolo,
    )
    
    tensors = pipeline.process_video(args.video, args.output)
    print(f"Generated {len(tensors)} feature tensors")
    
    if args.output:
        print(f"Saved to: {args.output}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Dashcam Collision Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "--model", "-m", default="EfficientNet-B0",
        help="Model name or 'all'"
    )
    train_parser.add_argument(
        "--epochs", "-e", type=int, default=50,
        help="Number of epochs"
    )
    train_parser.add_argument(
        "--batch-size", "-b", type=int, default=32,
        help="Batch size"
    )
    train_parser.add_argument(
        "--loss", "-l", default="bce", choices=["bce", "focal"],
        help="Loss function"
    )
    train_parser.set_defaults(func=cmd_train)
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export model to ONNX")
    export_parser.add_argument(
        "--model", "-m", default="EfficientNet-B0",
        help="Model name or 'all'"
    )
    export_parser.add_argument(
        "--format", "-f", nargs="+", default=["fp32", "fp16"],
        choices=["fp32", "fp16", "int8"],
        help="Export formats"
    )
    export_parser.set_defaults(func=cmd_export)
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model")
    eval_parser.add_argument(
        "--model", "-m", default="EfficientNet-B0",
        help="Model name or 'all'"
    )
    eval_parser.add_argument(
        "--batch-size", "-b", type=int, default=32,
        help="Batch size"
    )
    eval_parser.set_defaults(func=cmd_evaluate)
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available models")
    list_parser.set_defaults(func=cmd_list)
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess video")
    preprocess_parser.add_argument("video", help="Path to input video")
    preprocess_parser.add_argument(
        "--output", "-o", default=None,
        help="Output directory for tensors"
    )
    preprocess_parser.add_argument(
        "--sample-rate", "-s", type=int, default=15,
        help="Frame sampling rate"
    )
    preprocess_parser.add_argument(
        "--no-yolo", action="store_true",
        help="Disable YOLO vehicle detection"
    )
    preprocess_parser.set_defaults(func=cmd_preprocess)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    setup_logging(args.verbose)
    
    # Ensure directories exist
    from config.settings import ensure_dirs
    ensure_dirs()
    
    # Run command
    args.func(args)


if __name__ == "__main__":
    main()
