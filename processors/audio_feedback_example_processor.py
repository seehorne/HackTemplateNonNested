"""
Audio Feedback Example Processor

This processor demonstrates how to use the AudioFeedbackProcessor as a building block.
It shows various ways to integrate audio feedback into other processors for accessibility.

Example use case: Object proximity detector with audio feedback.
"""

from .base_processor import BaseProcessor
from .audio_feedback_processor import AudioFeedbackProcessor
from ultralytics import YOLO
import numpy as np
import cv2
from typing import Dict, Union, Tuple, Optional
import math


class AudioFeedbackExampleProcessor(BaseProcessor):
    """
    Example processor showing how to integrate AudioFeedbackProcessor as a building block.
    Detects objects and provides audio feedback based on object proximity and centering.
    """
    
    def __init__(self, model_path="./models/yolo11n-seg.pt"):
        """
        Initialize Example Processor
        
        Args:
            model_path (str): Path to YOLO model weights
        """
        super().__init__()
        
        # Initialize object detection
        self.model = YOLO(model_path)
        
        # Initialize audio feedback building block
        self.audio_feedback = AudioFeedbackProcessor()
        
        # State tracking
        self.last_status = None
    
    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Union[str, Dict]]:
        """
        Process frame to detect objects and provide audio feedback
        
        Args:
            frame (numpy.ndarray): Input frame to process
            
        Returns:
            tuple: (processed_frame, result_dict)
        """
        # Get frame dimensions
        height, width = frame.shape[:2]
        frame_center = (width / 2, height / 2)
        
        # Run object detection
        results = self.model(frame, verbose=False)
        result = results[0]
        
        # Find closest object to center
        closest_object = self._find_closest_object(result, frame_center, width, height)
        
        if closest_object is None:
            # No object detected - scanning audio
            audio_data = self.audio_feedback.generate_audio_feedback(
                preset=AudioFeedbackProcessor.PRESET_SCANNING
            )
            
            message = "Scanning for objects..."
            status = "scanning"
        else:
            # Object detected - provide feedback
            bbox = closest_object['bbox']
            class_name = closest_object['class_name']
            distance_ratio = closest_object['distance_from_center']
            
            # Calculate object center
            obj_center_x = (bbox[0] + bbox[2]) / 2
            obj_center_y = (bbox[1] + bbox[3]) / 2
            
            # Calculate offset from center (normalized to -1 to 1)
            offset_x = (obj_center_x - frame_center[0]) / (width / 2)
            offset_y = (obj_center_y - frame_center[1]) / (height / 2)
            
            # Calculate combined offset
            total_offset = math.sqrt(offset_x**2 + offset_y**2)
            
            # Generate audio based on alignment
            if total_offset < 0.15:
                # Well centered - success tone
                audio_data = self.audio_feedback.generate_alignment_feedback(
                    offset=total_offset,
                    aligned_threshold=0.15
                )
                message = f"{class_name} centered!"
                status = "centered"
            else:
                # Not centered - proximity feedback based on distance
                audio_data = self.audio_feedback.generate_proximity_feedback(
                    distance=distance_ratio,
                    min_distance=0.0,
                    max_distance=1.0
                )
                
                # Provide directional guidance
                directions = []
                if abs(offset_x) > 0.15:
                    directions.append("left" if offset_x > 0 else "right")
                if abs(offset_y) > 0.15:
                    directions.append("up" if offset_y > 0 else "down")
                
                direction_text = " and ".join(directions) if directions else "centered"
                message = f"{class_name} detected. Move camera {direction_text}."
                status = "adjusting"
            
            # Draw visual feedback
            frame = self._draw_feedback(frame, bbox, (obj_center_x, obj_center_y), 
                                       frame_center, status == "centered")
        
        # Build result payload
        result_payload = {
            "message": message,
            "status": status,
            "audio": audio_data,
            "instructions": "This example demonstrates audio feedback integration"
        }
        
        return frame, result_payload
    
    def _find_closest_object(
        self, 
        result, 
        frame_center: Tuple[float, float],
        width: int,
        height: int
    ) -> Optional[Dict]:
        """
        Find object closest to frame center
        
        Returns:
            Dict with object info or None
        """
        boxes = result.boxes
        
        if boxes is None or len(boxes) == 0:
            return None
        
        closest = None
        min_distance = float('inf')
        
        for box in boxes:
            bbox = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls)
            class_name = result.names[class_id]
            confidence = float(box.conf)
            
            # Skip low confidence detections
            if confidence < 0.5:
                continue
            
            # Calculate object center
            obj_center_x = (bbox[0] + bbox[2]) / 2
            obj_center_y = (bbox[1] + bbox[3]) / 2
            
            # Distance from frame center
            distance = math.sqrt(
                (obj_center_x - frame_center[0])**2 + 
                (obj_center_y - frame_center[1])**2
            )
            
            # Normalize distance
            max_distance = math.sqrt((width/2)**2 + (height/2)**2)
            normalized_distance = distance / max_distance
            
            if distance < min_distance:
                min_distance = distance
                closest = {
                    'bbox': bbox.tolist(),
                    'class_name': class_name,
                    'confidence': confidence,
                    'distance_from_center': normalized_distance
                }
        
        return closest
    
    def _draw_feedback(
        self, 
        frame: np.ndarray, 
        bbox: list,
        obj_center: Tuple[float, float],
        frame_center: Tuple[float, float],
        is_centered: bool
    ) -> np.ndarray:
        """
        Draw visual feedback on frame
        """
        # Choose color based on status
        color = (0, 255, 0) if is_centered else (0, 165, 255)
        
        # Draw bounding box
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw object center
        obj_center_int = (int(obj_center[0]), int(obj_center[1]))
        cv2.circle(frame, obj_center_int, 8, color, -1)
        
        # Draw frame center crosshair
        frame_center_int = (int(frame_center[0]), int(frame_center[1]))
        cv2.line(frame, 
                (frame_center_int[0] - 20, frame_center_int[1]),
                (frame_center_int[0] + 20, frame_center_int[1]),
                (255, 255, 255), 2)
        cv2.line(frame,
                (frame_center_int[0], frame_center_int[1] - 20),
                (frame_center_int[0], frame_center_int[1] + 20),
                (255, 255, 255), 2)
        
        return frame
    
    def process_pointcloud(self, point_cloud_data: Dict) -> Tuple[Optional[Dict], Union[str, Dict]]:
        """
        Point cloud processing not implemented
        """
        return point_cloud_data, {
            "message": "AudioFeedbackExampleProcessor does not process point clouds."
        }


processor = AudioFeedbackExampleProcessor()
app = processor.app
