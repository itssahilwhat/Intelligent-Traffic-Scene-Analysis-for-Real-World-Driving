"""
Vehicle Detection and Masking Module
=====================================

Detect vehicles using YOLOv8 segmentation and generate binary masks.

DOCUMENTATION:
- Model: YOLOv8n-seg (Ultralytics, nano variant)
  * Parameters: 3.4M
  * Input: (640, 640) RGB image
  * Output: Bounding boxes + segmentation masks

- COCO Vehicle Classes:
  * Class 2: car
  * Class 5: bus
  * Class 7: truck

- Mask Generation:
  * Binary mask where 1=vehicle pixel, 0=background
  * Multiple vehicles: Union of all masks (OR operation)
  * Morphology: MORPH_CLOSE to fill small gaps

- Confidence Threshold: 0.5 (empirically chosen)
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Check if ultralytics is available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("ultralytics not installed. Vehicle detection disabled.")


class VehicleDetector:
    """
    Detect vehicles and generate segmentation masks using YOLOv8.
    
    Args:
        model_name: YOLOv8 model variant (default: "yolov8n-seg.pt")
        confidence: Detection confidence threshold (default: 0.5)
        vehicle_classes: COCO class IDs for vehicles (default: car, bus, truck)
        device: 'cuda', 'cpu', or 'auto'
        
    Example:
        >>> detector = VehicleDetector()
        >>> mask = detector.detect(frame)  # (H, W) binary mask
        >>> masked_frame = frame * mask[:, :, np.newaxis]
    """
    
    # COCO class mappings for vehicles
    VEHICLE_CLASSES = {
        2: "car",
        5: "bus",
        7: "truck",
        3: "motorcycle",
        1: "bicycle",
    }
    
    def __init__(
        self,
        model_name: str = "yolov8n-seg.pt",
        confidence: float = 0.5,
        vehicle_classes: Tuple[int, ...] = (2, 5, 7),
        device: str = "auto",
        apply_morphology: bool = True,
        kernel_size: int = 5,
    ):
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics package required. Install: pip install ultralytics")
        
        self.model_name = model_name
        self.confidence = confidence
        self.vehicle_classes = vehicle_classes
        self.apply_morphology = apply_morphology
        self.kernel_size = kernel_size
        
        # Morphology kernel for closing gaps
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        
        # Load model
        self.model = YOLO(model_name)
        if device != "auto":
            self.model.to(device)
        
        logger.info(f"Loaded {model_name} for vehicle detection")
    
    def detect(
        self,
        frame: np.ndarray,
        return_boxes: bool = False,
    ) -> np.ndarray:
        """
        Detect vehicles and return binary mask.
        
        Args:
            frame: Input image (H, W, 3) RGB format
            return_boxes: Also return bounding boxes
            
        Returns:
            Binary mask (H, W) where 1=vehicle, 0=background
            If return_boxes=True: Tuple of (mask, boxes)
        """
        h, w = frame.shape[:2]
        
        # Run inference
        results = self.model(frame, conf=self.confidence, verbose=False)
        
        # Initialize empty mask
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        boxes = []
        
        for result in results:
            if result.masks is None:
                continue
            
            for i, cls_id in enumerate(result.boxes.cls.cpu().numpy()):
                cls_id = int(cls_id)
                
                # Filter for vehicle classes only
                if cls_id not in self.vehicle_classes:
                    continue
                
                # Get segmentation mask
                mask = result.masks.data[i].cpu().numpy()
                
                # Resize mask to original frame size
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                mask = (mask > 0.5).astype(np.uint8)
                
                # Union with existing mask
                combined_mask = np.maximum(combined_mask, mask)
                
                # Store bounding box if requested
                if return_boxes:
                    box = result.boxes.xyxy[i].cpu().numpy()
                    conf = result.boxes.conf[i].cpu().numpy()
                    boxes.append({
                        "xyxy": box,
                        "class": cls_id,
                        "class_name": self.VEHICLE_CLASSES.get(cls_id, "unknown"),
                        "confidence": float(conf),
                    })
        
        # Apply morphological closing to fill gaps
        if self.apply_morphology and combined_mask.any():
            combined_mask = cv2.morphologyEx(
                combined_mask, cv2.MORPH_CLOSE, self.kernel, iterations=2
            )
        
        if return_boxes:
            return combined_mask, boxes
        return combined_mask
    
    def detect_sequence(
        self,
        frames: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Detect vehicles in a sequence of frames.
        
        Args:
            frames: List of input frames
            
        Returns:
            List of binary masks
        """
        masks = []
        for frame in frames:
            mask = self.detect(frame)
            masks.append(mask)
        return masks
    
    def visualize(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.5,
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """
        Overlay mask on frame for visualization.
        
        Args:
            frame: Original frame (H, W, 3)
            mask: Binary mask (H, W)
            alpha: Blend factor
            color: Overlay color (R, G, B)
            
        Returns:
            Visualization image (H, W, 3)
        """
        vis = frame.copy()
        
        # Create colored overlay
        overlay = np.zeros_like(frame)
        overlay[mask > 0] = color
        
        # Blend
        vis = cv2.addWeighted(vis, 1.0, overlay, alpha, 0)
        
        return vis


class SimpleMaskGenerator:
    """
    Fallback mask generator when YOLO is not available.
    Uses simple motion-based segmentation.
    """
    
    def __init__(self, threshold: float = 25.0):
        self.threshold = threshold
        self.prev_frame = None
    
    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Generate mask based on frame difference."""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return np.zeros(gray.shape, dtype=np.uint8)
        
        # Frame difference
        diff = cv2.absdiff(self.prev_frame, gray)
        mask = (diff > self.threshold).astype(np.uint8)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        self.prev_frame = gray
        return mask


def create_vehicle_detector(use_yolo: bool = True, **kwargs) -> VehicleDetector:
    """
    Factory function to create appropriate detector.
    
    Args:
        use_yolo: Use YOLO if available
        **kwargs: Arguments for VehicleDetector
        
    Returns:
        Detector instance
    """
    if use_yolo and YOLO_AVAILABLE:
        return VehicleDetector(**kwargs)
    else:
        logger.warning("Using SimpleMaskGenerator fallback")
        return SimpleMaskGenerator()
