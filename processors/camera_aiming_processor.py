"""
Camera Aiming Building Block Processor

This processor helps users who are blind or low vision to aim their camera properly
to capture photos or center relevant objects. It provides non-visual audio cues for:
- Directional guidance (move left/right/up/down)
- Centering confirmation (object is centered)
- Distance feedback (move closer/farther)
- Object size feedback (object fills frame appropriately)

This is designed as a building block that other processors can call or augment.
"""

from .base_processor import BaseProcessor
from ultralytics import YOLO
import numpy as np
import cv2
from typing import Dict, Union, Tuple, List, Optional
import math

class CameraAimingProcessor(BaseProcessor):
    """
    Camera aiming building block processor for accessibility.
    Provides non-visual feedback to help users center objects in frame.
    """
    
    # Centering thresholds
    CENTER_THRESHOLD = 0.15  # Object center must be within 15% of frame center
    SIZE_MIN_THRESHOLD = 0.08  # Object should occupy at least 8% of frame
    SIZE_MAX_THRESHOLD = 0.85  # Object should not exceed 85% of frame
    SIZE_OPTIMAL_MIN = 0.15  # Optimal size range: 15-60% of frame (allows more comfortable distance)
    SIZE_OPTIMAL_MAX = 0.60
    
    def __init__(self, 
                 model_path="./models/yolo11n-seg.pt",
                 target_class=None,
                 confidence_threshold=0.5):
        """
        Initialize Camera Aiming Processor
        
        Args:
            model_path (str): Path to YOLO model weights
            target_class (str or list): Specific class(es) to aim at. If None, uses largest object.
            confidence_threshold (float): Minimum confidence for detection
        """
        super().__init__()
        self.model = YOLO(model_path)
        self.target_class = target_class if isinstance(target_class, list) else ([target_class] if target_class else None)
        self.confidence_threshold = confidence_threshold
        
    def _get_guidance_internal(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Internal method to get full guidance dictionary
        
        Args:
            frame (numpy.ndarray): Input frame to process
            
        Returns:
            tuple: (processed_frame, guidance_dict)
                - processed_frame: Frame with visual indicators (optional)
                - guidance_dict: Dictionary with full aiming guidance
        """
        # Get frame dimensions
        height, width = frame.shape[:2]
        frame_center = (width / 2, height / 2)
        
        # Run object detection
        results = self.model(frame, verbose=False)
        result = results[0]
        
        # Find target object
        target_detection = self._find_target_object(result, frame_center, width, height)
        
        if target_detection is None:
            # No object detected
            guidance = {
                "status": "no_object",
                "message": "No object detected. Pan camera slowly to find target.",
                "audio_cue": "scanning",
                "centered": False,
                "object_name": None
            }
            return frame, guidance
        
        # Extract target information
        bbox = target_detection['bbox']
        class_name = target_detection['class_name']
        confidence = target_detection['confidence']
        
        # Calculate object center and size
        obj_center_x = (bbox[0] + bbox[2]) / 2
        obj_center_y = (bbox[1] + bbox[3]) / 2
        obj_width = bbox[2] - bbox[0]
        obj_height = bbox[3] - bbox[1]
        obj_area = obj_width * obj_height
        frame_area = width * height
        size_ratio = obj_area / frame_area
        
        # Calculate offset from center (normalized to -1 to 1)
        offset_x = (obj_center_x - frame_center[0]) / (width / 2)
        offset_y = (obj_center_y - frame_center[1]) / (height / 2)
        
        # Generate guidance
        guidance = self._generate_guidance(
            offset_x, offset_y, size_ratio, class_name, confidence
        )
        
        # Create visual feedback (optional overlay)
        output_frame = self._draw_guidance_overlay(
            frame.copy(), bbox, (obj_center_x, obj_center_y), 
            frame_center, guidance['centered']
        )
        
        return output_frame, guidance
    
    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Union[str, Dict]]:
        """
        Process frame to provide camera aiming guidance
        
        Args:
            frame (numpy.ndarray): Input frame to process
            
        Returns:
            tuple: (processed_frame, message_string)
                - processed_frame: Frame with visual indicators (optional)
                - message_string: Simple message for audio output (not full dict)
        """
        output_frame, guidance = self._get_guidance_internal(frame)
        # Return just the message string for clean audio output
        return output_frame, guidance['message']
    
    def _find_target_object(self, result, frame_center, width, height) -> Optional[Dict]:
        """
        Find the target object to aim at based on criteria
        
        Args:
            result: YOLO detection result
            frame_center: Tuple of (x, y) frame center
            width: Frame width
            height: Frame height
            
        Returns:
            Dict with bbox, class_name, confidence, or None if no suitable target
        """
        boxes = result.boxes
        
        if boxes is None or len(boxes) == 0:
            return None
        
        candidates = []
        
        for i, box in enumerate(boxes):
            bbox = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls)
            confidence = float(box.conf)
            class_name = result.names[class_id]
            
            # Filter by confidence
            if confidence < self.confidence_threshold:
                continue
            
            # Filter by target class if specified
            if self.target_class and class_name not in self.target_class:
                continue
            
            # Calculate object metrics
            obj_center_x = (bbox[0] + bbox[2]) / 2
            obj_center_y = (bbox[1] + bbox[3]) / 2
            obj_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            
            # Distance from center (for prioritization)
            dist_from_center = math.sqrt(
                (obj_center_x - frame_center[0])**2 + 
                (obj_center_y - frame_center[1])**2
            )
            
            candidates.append({
                'bbox': bbox.tolist(),
                'class_name': class_name,
                'confidence': confidence,
                'area': obj_area,
                'distance_from_center': dist_from_center
            })
        
        if not candidates:
            return None
        
        # Priority: If target class specified, choose largest of that class
        # Otherwise, choose object closest to center with reasonable size
        if self.target_class:
            # Sort by area (largest first)
            candidates.sort(key=lambda x: x['area'], reverse=True)
        else:
            # Sort by proximity to center
            candidates.sort(key=lambda x: x['distance_from_center'])
        
        return candidates[0]
    
    def _generate_guidance(self, offset_x: float, offset_y: float, 
                          size_ratio: float, class_name: str, 
                          confidence: float) -> Dict:
        """
        Generate aiming guidance based on object position and size
        
        Args:
            offset_x: Horizontal offset from center (-1 to 1)
            offset_y: Vertical offset from center (-1 to 1)
            size_ratio: Object size as ratio of frame (0 to 1)
            class_name: Name of detected object
            confidence: Detection confidence
            
        Returns:
            Dictionary with guidance information
        """
        # Check if centered
        is_centered = (abs(offset_x) < self.CENTER_THRESHOLD and 
                      abs(offset_y) < self.CENTER_THRESHOLD)
        
        # Check size
        size_ok = (self.SIZE_OPTIMAL_MIN <= size_ratio <= self.SIZE_OPTIMAL_MAX)
        too_close = size_ratio > self.SIZE_MAX_THRESHOLD
        too_far = size_ratio < self.SIZE_MIN_THRESHOLD
        
        # Build guidance message
        if is_centered and size_ok:
            message = f"{class_name} centered and sized perfectly. Ready to capture!"
            audio_cue = "perfect"
            status = "perfect"
        elif is_centered:
            if too_close:
                message = f"{class_name} centered but too close. Move camera back."
                audio_cue = "move_back"
                status = "too_close"
            elif too_far:
                message = f"{class_name} centered but too far. Move camera closer."
                audio_cue = "move_forward"
                status = "too_far"
            else:
                message = f"{class_name} centered. Good alignment!"
                audio_cue = "centered"
                status = "centered"
        else:
            # Provide directional guidance
            directions = []
            
            if abs(offset_x) > self.CENTER_THRESHOLD:
                if offset_x > 0:
                    directions.append("left")  # Object is to the right, move camera left
                else:
                    directions.append("right")  # Object is to the left, move camera right
            
            if abs(offset_y) > self.CENTER_THRESHOLD:
                if offset_y > 0:
                    directions.append("up")  # Object is below, move camera up
                else:
                    directions.append("down")  # Object is above, move camera down
            
            direction_text = " and ".join(directions)
            message = f"Move camera {direction_text} to center {class_name}."
            audio_cue = f"move_{'_'.join(directions)}"
            status = "adjusting"
        
        # Calculate distance intensity (for audio volume/frequency modulation)
        distance_from_center = math.sqrt(offset_x**2 + offset_y**2)
        
        return {
            "status": status,
            "message": message,
            "audio_cue": audio_cue,
            "centered": is_centered,
            "size_ok": size_ok,
            "object_name": class_name,
            "confidence": round(confidence, 2),
            "offset_x": round(offset_x, 2),
            "offset_y": round(offset_y, 2),
            "size_ratio": round(size_ratio, 2),
            "distance_from_center": round(distance_from_center, 2),
            "too_close": too_close,
            "too_far": too_far
        }
    
    def _draw_guidance_overlay(self, frame: np.ndarray, bbox: List[float],
                              obj_center: Tuple[float, float],
                              frame_center: Tuple[float, float],
                              is_centered: bool) -> np.ndarray:
        """
        Draw visual guidance overlay on frame
        
        Args:
            frame: Input frame
            bbox: Bounding box [x1, y1, x2, y2]
            obj_center: Object center (x, y)
            frame_center: Frame center (x, y)
            is_centered: Whether object is centered
            
        Returns:
            Frame with overlay
        """
        # Choose color based on centering status
        color = (0, 255, 0) if is_centered else (0, 165, 255)  # Green if centered, orange otherwise
        
        # Draw bounding box
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw object center
        obj_center_int = (int(obj_center[0]), int(obj_center[1]))
        cv2.circle(frame, obj_center_int, 8, color, -1)
        
        # Draw frame center crosshair
        frame_center_int = (int(frame_center[0]), int(frame_center[1]))
        crosshair_size = 30
        cv2.line(frame, 
                (frame_center_int[0] - crosshair_size, frame_center_int[1]),
                (frame_center_int[0] + crosshair_size, frame_center_int[1]),
                (255, 255, 255), 2)
        cv2.line(frame,
                (frame_center_int[0], frame_center_int[1] - crosshair_size),
                (frame_center_int[0], frame_center_int[1] + crosshair_size),
                (255, 255, 255), 2)
        
        # Draw line from object to center
        cv2.line(frame, obj_center_int, frame_center_int, color, 2)
        
        return frame
    
    def process_pointcloud(self, point_cloud_data: Dict) -> Tuple[Optional[Dict], Union[str, Dict]]:
        """
        Point cloud processing not implemented for camera aiming
        """
        return point_cloud_data, {"message": "CameraAimingProcessor does not process point clouds."}
    
    def get_aiming_guidance(self, frame: np.ndarray, target_class: str = None) -> Dict:
        """
        Helper method for other processors to get full aiming guidance dictionary
        
        Args:
            frame: Input frame
            target_class: Optional specific class to target
            
        Returns:
            Dictionary with full aiming guidance (status, message, centered, etc.)
        """
        # Temporarily override target class if specified
        original_target = self.target_class
        if target_class:
            self.target_class = [target_class]
        
        # Get full guidance using internal method
        _, guidance = self._get_guidance_internal(frame)
        
        # Restore original target
        self.target_class = original_target
        
        return guidance


processor = CameraAimingProcessor()
app = processor.app
