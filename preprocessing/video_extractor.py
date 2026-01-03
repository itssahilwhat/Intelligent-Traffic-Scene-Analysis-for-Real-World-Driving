"""
Video Frame Extraction Module
=============================

Extract frames from dashcam videos using OpenCV.

DOCUMENTATION:
- Library: OpenCV 4.x (cv2.VideoCapture)
- Frame sampling: Extract every Nth frame from video
- Example: For 30 FPS input with N=15 → 2 FPS output
- Output: NumPy arrays with shape (H, W, 3) in BGR format
"""

import logging
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoExtractor:
    """
    Extract frames from video files.
    
    Args:
        sample_rate: Extract every N frames (default: 15 for 30fps→2fps)
        target_size: Optional resize (width, height)
        
    Example:
        >>> extractor = VideoExtractor(sample_rate=15)
        >>> frames = extractor.extract("video.mp4")
        >>> print(f"Extracted {len(frames)} frames")
    """
    
    def __init__(
        self,
        sample_rate: int = 15,
        target_size: Optional[Tuple[int, int]] = None,
    ):
        self.sample_rate = sample_rate
        self.target_size = target_size
    
    def extract(self, video_path: str) -> List[np.ndarray]:
        """
        Extract all sampled frames from video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of frames as numpy arrays (H, W, 3) in RGB format
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            RuntimeError: If video cannot be opened
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        frames = list(self.extract_generator(video_path))
        logger.info(f"Extracted {len(frames)} frames from {path.name}")
        return frames
    
    def extract_generator(self, video_path: str) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields frames one at a time (memory efficient).
        
        Args:
            video_path: Path to video file
            
        Yields:
            Frames as numpy arrays (H, W, 3) in RGB format
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        try:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample every Nth frame
                if frame_idx % self.sample_rate == 0:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Resize if specified
                    if self.target_size is not None:
                        frame = cv2.resize(
                            frame, 
                            self.target_size, 
                            interpolation=cv2.INTER_LINEAR
                        )
                    
                    yield frame
                
                frame_idx += 1
        finally:
            cap.release()
    
    def get_video_info(self, video_path: str) -> dict:
        """
        Get video metadata.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with fps, width, height, frame_count, duration
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        try:
            info = {
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
            }
            info["duration"] = info["frame_count"] / info["fps"] if info["fps"] > 0 else 0
            return info
        finally:
            cap.release()
    
    def validate_video(self, video_path: str) -> Tuple[bool, str]:
        """
        Check if video can be processed.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        path = Path(video_path)
        
        if not path.exists():
            return False, "File not found"
        
        try:
            info = self.get_video_info(video_path)
            
            if info["frame_count"] == 0:
                return False, "Video has no frames"
            
            if info["fps"] <= 0:
                return False, "Invalid FPS"
            
            if info["width"] == 0 or info["height"] == 0:
                return False, "Invalid resolution"
            
            return True, "OK"
            
        except Exception as e:
            return False, str(e)


def extract_frames(
    video_path: str,
    output_dir: str,
    sample_rate: int = 15,
    save_format: str = "npy",
) -> List[str]:
    """
    Extract frames and save to disk.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save frames
        sample_rate: Extract every N frames
        save_format: 'npy' for numpy, 'png' for images
        
    Returns:
        List of saved file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    extractor = VideoExtractor(sample_rate=sample_rate)
    frames = extractor.extract(video_path)
    
    saved_paths = []
    for idx, frame in enumerate(frames):
        if save_format == "npy":
            file_path = output_path / f"frame_{idx:04d}.npy"
            np.save(file_path, frame)
        else:
            file_path = output_path / f"frame_{idx:04d}.png"
            # Convert RGB back to BGR for OpenCV saving
            cv2.imwrite(str(file_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        saved_paths.append(str(file_path))
    
    logger.info(f"Saved {len(saved_paths)} frames to {output_dir}")
    return saved_paths
