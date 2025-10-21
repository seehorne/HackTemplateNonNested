"""
Object Finder Processor for Blind Users

This processor helps blind users locate and reach objects using non-visual audio cues.
It combines hand tracking and object detection to guide users to pick up objects.

Key Features:
- Detects objects in the camera view using YOLO
- Tracks user's hand position using MediaPipe
- Provides directional audio guidance (left, right, up, down)
- Gives distance feedback (closer, reached)
- Uses spatial audio panning for enhanced directional cues
- CPU-only, no GPU required

This processor leverages existing building blocks:
- HandTrackingProcessor for hand detection
- YOLOProcessor for object detection
"""

from .base_processor import BaseProcessor
from .hand_tracking_processor import HandTrackingProcessor
from .scene_object_processor import YOLOProcessor
import numpy as np
import cv2
from typing import Dict, Union, Tuple, List, Optional
import math


class ObjectFinderProcessor(BaseProcessor):
    """
    Processor that guides blind users to find and reach objects using their hands.
    Combines hand tracking and object detection with audio guidance.
    """
    
    def __init__(self, 
                 model_path="./models/yolo11n-seg.pt",
                 target_object_class=None,
                 guidance_threshold=50):
        """
        Initialize Object Finder Processor
        
        Args:
            model_path (str): Path to YOLO model weights
            target_object_class (str): Specific object class to find (None = any object)
            guidance_threshold (int): Distance in pixels to consider "reached" (default: 50)
        """
        super().__init__()
        
        # Initialize building block processors
        self.hand_tracker = HandTrackingProcessor(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=1  # Only track one hand for simplicity
        )
        self.object_detector = YOLOProcessor(model_path=model_path)
        
        # Configuration
        self.target_object_class = target_object_class
        self.guidance_threshold = guidance_threshold
        
        # State tracking
        self.last_guidance_message = ""
        self.target_object = None  # Currently targeted object
        
    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Union[str, Dict]]:
        """
        Process frame to guide user to find objects with their hand
        
        Args:
            frame (numpy.ndarray): Input frame to process
            
        Returns:
            tuple: (processed_frame, guidance_message)
                - processed_frame: Frame with visual indicators (optional)
                - guidance_message: Simple message string for audio output
        """
        height, width = frame.shape[:2]
        output_frame = frame.copy()
        
        # Step 1: Detect objects in the scene with bounding boxes
        detected_objects = self._get_object_detections(frame)
        
        # Step 2: Get hand tracking data
        hand_data = self.hand_tracker.get_hand_tracking_data(frame)
        
        # Step 3: Generate guidance based on hand position and objects
        guidance_message = self._generate_guidance(
            hand_data, 
            detected_objects, 
            width, 
            height
        )
        
        # Step 4: Draw visual indicators (for debugging/sighted helpers)
        output_frame = self._draw_visual_feedback(
            output_frame,
            hand_data,
            detected_objects,
            width,
            height
        )
        
        # Return just the message string for clean audio output
        return output_frame, guidance_message
    

    
    def _get_object_detections(self, frame: np.ndarray) -> List[Dict]:
        """
        Get detailed object detections with bounding boxes using YOLO directly
        
        Returns:
            List of dictionaries with bbox, class_name, confidence, center
        """
        results = self.object_detector.model(frame)
        result = results[0]
        
        detections = []
        boxes = result.boxes
        
        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                box = boxes[i]
                bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                class_id = int(box.cls)
                confidence = float(box.conf)
                class_name = result.names[class_id]
                
                # Calculate center of bounding box
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                
                detection = {
                    'bbox': bbox,
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'center': (int(center_x), int(center_y))
                }
                
                # Filter by target class if specified
                if self.target_object_class is None or class_name == self.target_object_class:
                    detections.append(detection)
        
        return detections
    
    def _generate_guidance(self, 
                          hand_data: Dict, 
                          detected_objects: List[Dict],
                          frame_width: int,
                          frame_height: int) -> str:
        """
        Generate audio guidance message
        
        Args:
            hand_data: Hand tracking data from HandTrackingProcessor
            detected_objects: List of detected objects with positions
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            
        Returns:
            guidance_message: Text message for TTS
        """
        
        # Check if hand is detected
        if hand_data['hand_count'] == 0:
            # List detected objects if any
            if len(detected_objects) > 0:
                object_names = [obj['class_name'] for obj in detected_objects]
                unique_objects = list(set(object_names))
                if len(unique_objects) == 1:
                    return f"Show your hand to find the {unique_objects[0]}."
                elif len(unique_objects) <= 3:
                    objects_str = ", ".join(unique_objects)
                    return f"Show your hand to find objects. I see: {objects_str}."
                else:
                    return f"Show your hand to find objects. I see {len(unique_objects)} different objects."
            return "Show your hand to start finding objects."
        
        # Check if objects are detected
        if len(detected_objects) == 0:
            return "No objects detected. Move camera to scan the area."
        
        # Get hand position (use index finger tip or hand center)
        hand = hand_data['hands'][0]
        hand_pixel_x = hand['center']['pixel_x']
        hand_pixel_y = hand['center']['pixel_y']
        
        # Find closest object to hand and get info about nearby objects
        closest_object = None
        min_distance = float('inf')
        nearby_objects = []  # Objects within reasonable distance
        
        for obj in detected_objects:
            obj_center = obj['center']
            distance = math.sqrt(
                (hand_pixel_x - obj_center[0])**2 + 
                (hand_pixel_y - obj_center[1])**2
            )
            
            # Track closest object
            if distance < min_distance:
                min_distance = distance
                closest_object = obj
            
            # Track nearby objects (within 2x the guidance threshold)
            if distance < self.guidance_threshold * 3:
                nearby_objects.append({
                    'object': obj,
                    'distance': distance
                })
        
        if closest_object is None:
            return "Objects detected. Show your hand to locate them."
        
        # Sort nearby objects by distance
        nearby_objects.sort(key=lambda x: x['distance'])
        
        # Calculate direction and distance to closest object
        obj_center = closest_object['center']
        dx = obj_center[0] - hand_pixel_x
        dy = obj_center[1] - hand_pixel_y
        
        # Check if hand has reached the object
        if min_distance < self.guidance_threshold:
            message = f"Object reached! {closest_object['class_name']} is right there."
            # Add info about other nearby objects if any
            other_nearby = [n for n in nearby_objects if n['object']['class_name'] != closest_object['class_name']]
            if len(other_nearby) > 0:
                other_names = [n['object']['class_name'] for n in other_nearby[:2]]  # Max 2 others
                message += f" Also nearby: {', '.join(other_names)}."
            return message
        
        # Generate directional guidance
        direction_parts = []
        
        # Horizontal direction
        if abs(dx) > 30:  # Threshold to avoid jittery instructions
            if dx > 0:
                direction_parts.append("right")
            else:
                direction_parts.append("left")
        
        # Vertical direction
        if abs(dy) > 30:
            if dy > 0:
                direction_parts.append("down")
            else:
                direction_parts.append("up")
        
        # Build message with object identification
        distance_description = self._describe_distance(min_distance, frame_width, frame_height)
        
        if len(direction_parts) == 0:
            message = f"Almost there! {closest_object['class_name']} is very close."
        else:
            direction_str = " and ".join(direction_parts)
            message = f"Move hand {direction_str}. {closest_object['class_name']} is {distance_description}."
        
        # Add info about other nearby objects if there are multiple close by
        if len(nearby_objects) > 1:
            other_objects = [n['object']['class_name'] for n in nearby_objects[1:3] if n['object']['class_name'] != closest_object['class_name']]
            if other_objects:
                unique_others = list(set(other_objects))
                if len(unique_others) == 1:
                    message += f" {unique_others[0]} also nearby."
                else:
                    message += f" Also nearby: {', '.join(unique_others[:2])}."
        
        return message
    
    def _describe_distance(self, distance: float, frame_width: int, frame_height: int) -> str:
        """
        Convert pixel distance to user-friendly description
        
        Args:
            distance: Distance in pixels
            frame_width: Frame width
            frame_height: Frame height
            
        Returns:
            String description of distance
        """
        # Normalize distance relative to frame diagonal
        frame_diagonal = math.sqrt(frame_width**2 + frame_height**2)
        normalized_distance = distance / frame_diagonal
        
        if normalized_distance < 0.1:
            return "very close"
        elif normalized_distance < 0.2:
            return "close"
        elif normalized_distance < 0.35:
            return "medium distance"
        else:
            return "far away"
    
    def _draw_visual_feedback(self,
                             frame: np.ndarray,
                             hand_data: Dict,
                             detected_objects: List[Dict],
                             frame_width: int,
                             frame_height: int) -> np.ndarray:
        """
        Draw visual indicators on frame (for debugging/sighted helpers)
        
        Args:
            frame: Input frame
            hand_data: Hand tracking data
            detected_objects: Detected objects
            frame_width: Frame width
            frame_height: Frame height
            
        Returns:
            Frame with visual overlays
        """
        output = frame.copy()
        
        # Draw object bounding boxes
        for obj in detected_objects:
            bbox = obj['bbox']
            center = obj['center']
            
            # Draw bounding box
            cv2.rectangle(
                output,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                (0, 255, 0),  # Green
                2
            )
            
            # Draw center point
            cv2.circle(output, center, 8, (0, 255, 0), -1)
            
            # Draw label
            label = f"{obj['class_name']} {obj['confidence']:.2f}"
            cv2.putText(
                output,
                label,
                (int(bbox[0]), int(bbox[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
        
        # Draw hand position
        if hand_data['hand_count'] > 0:
            hand = hand_data['hands'][0]
            hand_pos = (hand['center']['pixel_x'], hand['center']['pixel_y'])
            
            # Draw hand center
            cv2.circle(output, hand_pos, 12, (255, 255, 0), -1)  # Yellow circle
            
            # Draw line to closest object
            if len(detected_objects) > 0:
                closest_obj = min(
                    detected_objects,
                    key=lambda obj: math.sqrt(
                        (hand_pos[0] - obj['center'][0])**2 + 
                        (hand_pos[1] - obj['center'][1])**2
                    )
                )
                
                # Draw connecting line
                cv2.line(
                    output,
                    hand_pos,
                    closest_obj['center'],
                    (255, 0, 255),  # Magenta
                    2
                )
                
                # Draw distance text
                distance = math.sqrt(
                    (hand_pos[0] - closest_obj['center'][0])**2 + 
                    (hand_pos[1] - closest_obj['center'][1])**2
                )
                mid_point = (
                    (hand_pos[0] + closest_obj['center'][0]) // 2,
                    (hand_pos[1] + closest_obj['center'][1]) // 2
                )
                cv2.putText(
                    output,
                    f"{int(distance)}px",
                    mid_point,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
        
        return output
    
    def process_pointcloud(self, point_cloud_data: Dict) -> Tuple[Optional[Dict], Union[str, Dict]]:
        """
        Point cloud processing not implemented for object finder
        """
        return point_cloud_data, {
            "message": "ObjectFinderProcessor does not process point clouds."
        }


# Create processor instance
processor = ObjectFinderProcessor()
app = processor.app
