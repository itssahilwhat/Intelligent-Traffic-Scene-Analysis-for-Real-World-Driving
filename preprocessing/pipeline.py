"""
End-to-End Preprocessing Pipeline
==================================

Complete pipeline from raw video to 6-channel tensors.

Usage:
    >>> pipeline = PreprocessingPipeline()
    >>> tensors = pipeline.process_video("dashcam.mp4")
    >>> print(f"Generated {len(tensors)} feature tensors")
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from .video_extractor import VideoExtractor
from .optical_flow import OpticalFlowComputer
from .vehicle_detector import VehicleDetector, SimpleMaskGenerator, YOLO_AVAILABLE
from .feature_assembler import FeatureAssembler

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline for dashcam collision prediction.
    
    Pipeline steps:
    1. Extract frames from video (every Nth frame)
    2. Compute optical flow between consecutive frames
    3. Detect vehicles and generate masks
    4. Assemble 6-channel feature tensors
    
    Args:
        frame_sample_rate: Extract every N frames (default: 15)
        use_yolo: Use YOLO for vehicle detection (default: True)
        target_size: Output tensor size (default: (224, 224))
        
    Example:
        >>> pipeline = PreprocessingPipeline()
        >>> tensors = pipeline.process_video("video.mp4")
    """
    
    def __init__(
        self,
        frame_sample_rate: int = 15,
        use_yolo: bool = True,
        target_size: tuple = (224, 224),
        yolo_model: str = "yolov8n-seg.pt",
        yolo_confidence: float = 0.5,
    ):
        self.frame_sample_rate = frame_sample_rate
        self.target_size = target_size
        
        # Initialize components
        self.video_extractor = VideoExtractor(sample_rate=frame_sample_rate)
        self.flow_computer = OpticalFlowComputer()
        self.feature_assembler = FeatureAssembler(target_size=target_size)
        
        # Vehicle detector (with fallback)
        if use_yolo and YOLO_AVAILABLE:
            self.vehicle_detector = VehicleDetector(
                model_name=yolo_model,
                confidence=yolo_confidence,
            )
        else:
            logger.warning("Using motion-based mask generator (YOLO not available)")
            self.vehicle_detector = SimpleMaskGenerator()
    
    def process_video(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
    ) -> List[torch.Tensor]:
        """
        Process a single video file.
        
        Args:
            video_path: Path to input video
            output_dir: Optional directory to save tensors
            
        Returns:
            List of 6-channel tensors, one per frame pair
        """
        logger.info(f"Processing video: {video_path}")
        
        # Step 1: Extract frames
        frames = self.video_extractor.extract(video_path)
        if len(frames) < 2:
            logger.warning(f"Video too short: {len(frames)} frames")
            return []
        
        logger.info(f"Extracted {len(frames)} frames")
        
        # Step 2: Compute optical flow
        flows = self.flow_computer.compute_sequence(frames, normalize=True)
        logger.info(f"Computed {len(flows)} flow fields")
        
        # Step 3: Detect vehicles
        masks = []
        for frame in frames[1:]:  # Align with flows (skip first frame)
            mask = self.vehicle_detector.detect(frame)
            masks.append(mask)
        
        logger.info(f"Generated {len(masks)} vehicle masks")
        
        # Step 4: Assemble 6-channel tensors
        tensors = []
        for i in range(len(flows)):
            # Use frame[i+1] (current frame), flow[i], mask[i]
            tensor = self.feature_assembler.assemble(
                rgb=frames[i + 1],
                flow=flows[i],
                mask=masks[i],
            )
            tensors.append(tensor)
        
        logger.info(f"Assembled {len(tensors)} feature tensors")
        
        # Save if output directory specified
        if output_dir:
            self._save_tensors(tensors, output_dir)
        
        return tensors
    
    def process_frames(
        self,
        frames: List[np.ndarray],
    ) -> List[torch.Tensor]:
        """
        Process a list of pre-extracted frames.
        
        Args:
            frames: List of RGB frames (H, W, 3)
            
        Returns:
            List of 6-channel tensors
        """
        if len(frames) < 2:
            return []
        
        # Compute flow
        flows = self.flow_computer.compute_sequence(frames, normalize=True)
        
        # Detect vehicles
        masks = [self.vehicle_detector.detect(f) for f in frames[1:]]
        
        # Assemble tensors
        tensors = []
        for i in range(len(flows)):
            tensor = self.feature_assembler.assemble(
                rgb=frames[i + 1],
                flow=flows[i],
                mask=masks[i],
            )
            tensors.append(tensor)
        
        return tensors
    
    def _save_tensors(self, tensors: List[torch.Tensor], output_dir: str) -> None:
        """Save tensors to disk."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        for idx, tensor in enumerate(tensors):
            torch.save(tensor, out_path / f"tensor_{idx:04d}.pt")
        
        logger.info(f"Saved {len(tensors)} tensors to {output_dir}")


def process_video_to_tensors(
    video_path: str,
    output_dir: str,
    frame_sample_rate: int = 15,
) -> List[str]:
    """
    Convenience function to process video and save tensors.
    
    Args:
        video_path: Input video path
        output_dir: Output directory
        frame_sample_rate: Frame sampling rate
        
    Returns:
        List of saved tensor paths
    """
    pipeline = PreprocessingPipeline(frame_sample_rate=frame_sample_rate)
    tensors = pipeline.process_video(video_path, output_dir)
    
    return [
        str(Path(output_dir) / f"tensor_{i:04d}.pt")
        for i in range(len(tensors))
    ]
