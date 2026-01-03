"""
Optical Flow Computation Module
===============================

Compute dense optical flow using Farneback algorithm.

DOCUMENTATION:
- Algorithm: Farneback (cv2.calcOpticalFlowFarneback)
- Why Farneback over alternatives:
  * Horn-Schunck: Global method, slower, less accurate on boundaries
  * Lucas-Kanade: Sparse (corner points only), not suitable for dense motion
  * Farneback: Dense, polynomial approximation, good balance of speed/accuracy

- Parameters selected based on typical dashcam footage:
  * pyr_scale=0.5: Standard pyramid decimation
  * levels=3: Sufficient for 720p video
  * winsize=15: Captures vehicle-scale motion
  * iterations=3: Convergence for smooth flow
  * poly_n=5: Neighborhood for polynomial expansion
  * poly_sigma=1.2: Smoothing for polynomial coefficients

- Input: Consecutive grayscale frames, uint8 [0-255]
- Output: Flow array (H, W, 2) where [Vx, Vy] in pixels/frame
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class OpticalFlowComputer:
    """
    Compute Farneback optical flow between consecutive frames.
    
    Args:
        pyr_scale: Pyramid scale factor (default: 0.5)
        levels: Number of pyramid levels (default: 3)
        winsize: Averaging window size (default: 15)
        iterations: Iterations per pyramid level (default: 3)
        poly_n: Polynomial neighborhood size (default: 5)
        poly_sigma: Gaussian std for polynomial smoothing (default: 1.2)
        max_flow: Clip flow magnitude (default: 20.0 pixels/frame)
        
    Example:
        >>> flow_computer = OpticalFlowComputer()
        >>> flow = flow_computer.compute(prev_frame, curr_frame)
        >>> print(f"Flow shape: {flow.shape}")  # (H, W, 2)
    """
    
    def __init__(
        self,
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
        max_flow: float = 20.0,
        use_gaussian: bool = False,
    ):
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.max_flow = max_flow
        
        # Flags for cv2.calcOpticalFlowFarneback
        self.flags = cv2.OPTFLOW_FARNEBACK_GAUSSIAN if use_gaussian else 0
    
    def compute(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Compute optical flow between two frames.
        
        Args:
            prev_frame: Previous frame (H, W, 3) RGB or (H, W) grayscale
            curr_frame: Current frame (H, W, 3) RGB or (H, W) grayscale
            normalize: Scale output to [0, 1] range (default: True)
            
        Returns:
            Flow array (H, W, 2) with [Vx, Vy] components
            If normalize=True: Values in [0, 1]
            If normalize=False: Raw pixel velocities clipped to [-max_flow, max_flow]
        """
        # Convert to grayscale if needed
        prev_gray = self._to_grayscale(prev_frame)
        curr_gray = self._to_grayscale(curr_frame)
        
        # Compute Farneback optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            flow=None,              # Output array
            pyr_scale=self.pyr_scale,
            levels=self.levels,
            winsize=self.winsize,
            iterations=self.iterations,
            poly_n=self.poly_n,
            poly_sigma=self.poly_sigma,
            flags=self.flags,
        )
        
        # Clip to maximum flow
        flow = np.clip(flow, -self.max_flow, self.max_flow)
        
        # Normalize to [0, 1] if requested
        if normalize:
            flow = (flow + self.max_flow) / (2 * self.max_flow)
        
        return flow.astype(np.float32)
    
    def compute_sequence(
        self,
        frames: list,
        normalize: bool = True,
    ) -> list:
        """
        Compute optical flow for a sequence of frames.
        
        Args:
            frames: List of frames (N frames → N-1 flow fields)
            normalize: Scale output to [0, 1] range
            
        Returns:
            List of flow arrays, one fewer than input frames
        """
        if len(frames) < 2:
            logger.warning("Need at least 2 frames for optical flow")
            return []
        
        flows = []
        for i in range(len(frames) - 1):
            flow = self.compute(frames[i], frames[i + 1], normalize=normalize)
            flows.append(flow)
        
        return flows
    
    def _to_grayscale(self, frame: np.ndarray) -> np.ndarray:
        """Convert frame to grayscale uint8."""
        if frame.ndim == 2:
            gray = frame
        elif frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame[:, :, 0]
        
        # Ensure uint8
        if gray.dtype != np.uint8:
            if gray.max() <= 1.0:
                gray = (gray * 255).astype(np.uint8)
            else:
                gray = gray.astype(np.uint8)
        
        return gray
    
    def visualize(
        self,
        flow: np.ndarray,
        scale: float = 1.0,
    ) -> np.ndarray:
        """
        Visualize optical flow using HSV color coding.
        
        Color coding:
        - Hue: Direction of motion (0-360 degrees)
        - Saturation: Always 255 (full)
        - Value: Magnitude of motion
        
        Args:
            flow: Flow array (H, W, 2)
            scale: Scale factor for magnitude visualization
            
        Returns:
            RGB image (H, W, 3) uint8 visualizing the flow
        """
        h, w = flow.shape[:2]
        
        # Get flow components (denormalize if needed)
        fx, fy = flow[:, :, 0], flow[:, :, 1]
        
        # If normalized [0,1], convert back to raw
        if fx.max() <= 1.0 and fx.min() >= 0.0:
            fx = fx * 2 * self.max_flow - self.max_flow
            fy = fy * 2 * self.max_flow - self.max_flow
        
        # Compute magnitude and angle
        magnitude = np.sqrt(fx**2 + fy**2)
        angle = np.arctan2(fy, fx)
        
        # Create HSV image
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[:, :, 0] = ((angle + np.pi) / (2 * np.pi) * 180).astype(np.uint8)  # Hue
        hsv[:, :, 1] = 255  # Saturation
        hsv[:, :, 2] = np.clip(magnitude * scale * 10, 0, 255).astype(np.uint8)  # Value
        
        # Convert to RGB
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return rgb


def compute_flow_for_video(
    frames: list,
    output_dir: Optional[str] = None,
    normalize: bool = True,
) -> list:
    """
    Convenience function to compute flow for all frame pairs.
    
    Args:
        frames: List of video frames
        output_dir: Optional directory to save flow arrays
        normalize: Normalize to [0, 1] range
        
    Returns:
        List of flow arrays
    """
    from pathlib import Path
    
    computer = OpticalFlowComputer()
    flows = computer.compute_sequence(frames, normalize=normalize)
    
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        for idx, flow in enumerate(flows):
            np.save(out_path / f"flow_{idx:04d}.npy", flow)
        
        logger.info(f"Saved {len(flows)} flow fields to {output_dir}")
    
    return flows
