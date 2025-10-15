"""
Hand Tracking Building Block Processor

This processor helps users who are blind or low vision to locate hands in their
camera view. It provides non-visual audio cues for:
- Hand presence/absence detection
- Spatial location (left/right/up/down quadrants)
- Hand count (how many hands detected)
- Distance estimation (near/far)

This is designed as a building block that other processors can call or augment.
Key features:
- CPU-only using MediaPipe Hands
- Returns structured data for other processors to use
- Provides simple audio-friendly messages
- Exposes detailed hand landmark data for advanced use cases
"""

from .base_processor import BaseProcessor
import mediapipe as mp
import numpy as np
import cv2
from typing import Dict, Union, Tuple, List, Optional

class HandTrackingProcessor(BaseProcessor):
    """
    Hand tracking building block processor for accessibility.
    Provides non-visual feedback about hand locations in frame.
    """
    
    def __init__(self, 
                 min_detection_confidence=0.7,
                 min_tracking_confidence=0.7,
                 max_num_hands=2):
        """
        Initialize Hand Tracking Processor
        
        Args:
            min_detection_confidence (float): Minimum confidence for hand detection
            min_tracking_confidence (float): Minimum confidence for hand tracking
            max_num_hands (int): Maximum number of hands to detect (1 or 2)
        """
        super().__init__()
        
        # Initialize MediaPipe hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            model_complexity=1,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
    def _get_hand_data_internal(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Internal method to get detailed hand tracking data
        
        Args:
            frame (numpy.ndarray): Input frame to process
            
        Returns:
            tuple: (processed_frame, hand_data_dict)
                - processed_frame: Frame with visual indicators (optional)
                - hand_data_dict: Dictionary with full hand tracking data
        """
        # Get frame dimensions
        height, width = frame.shape[:2]
        frame_center = (width / 2, height / 2)
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run hand detection
        results = self.hands.process(frame_rgb)
        
        # Create output frame
        output_frame = frame.copy()
        
        if not results.multi_hand_landmarks:
            # No hands detected
            hand_data = {
                "status": "no_hands",
                "message": "No hands detected in view.",
                "hand_count": 0,
                "hands": []
            }
            return output_frame, hand_data
        
        # Process detected hands
        hands_info = []
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Get handedness (left/right)
            handedness = "unknown"
            if results.multi_handedness:
                handedness = results.multi_handedness[hand_idx].classification[0].label
            
            # Calculate hand center (average of all landmarks)
            hand_center_x = np.mean([lm.x for lm in hand_landmarks.landmark])
            hand_center_y = np.mean([lm.y for lm in hand_landmarks.landmark])
            
            # Normalize to frame coordinates
            hand_center_px = (int(hand_center_x * width), int(hand_center_y * height))
            
            # Calculate hand size (bounding box area)
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            hand_width = (max(x_coords) - min(x_coords)) * width
            hand_height = (max(y_coords) - min(y_coords)) * height
            hand_area = hand_width * hand_height
            frame_area = width * height
            size_ratio = hand_area / frame_area
            
            # Calculate offset from center (normalized to -1 to 1)
            offset_x = (hand_center_x * width - frame_center[0]) / (width / 2)
            offset_y = (hand_center_y * height - frame_center[1]) / (height / 2)
            
            # Determine spatial location
            location = self._determine_location(offset_x, offset_y)
            
            # Determine distance and beep parameters
            distance, beep_params = self._determine_distance(size_ratio)
            
            # Draw hand landmarks on output frame
            self.mp_drawing.draw_landmarks(
                output_frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Draw hand center indicator
            cv2.circle(output_frame, hand_center_px, 10, (0, 255, 255), -1)
            
            # Store hand information
            hand_info = {
                "hand_index": hand_idx,
                "handedness": handedness,
                "center": {
                    "x": round(hand_center_x, 3),
                    "y": round(hand_center_y, 3),
                    "pixel_x": hand_center_px[0],
                    "pixel_y": hand_center_px[1]
                },
                "offset": {
                    "x": round(offset_x, 2),
                    "y": round(offset_y, 2)
                },
                "location": location,
                "distance": distance,
                "beep_params": beep_params,
                "size_ratio": round(size_ratio, 3),
                "landmarks": hand_landmarks  # Full landmark data for advanced use
            }
            hands_info.append(hand_info)
        
        # Generate overall message
        message = self._generate_message(hands_info)
        
        # Build complete hand data dictionary
        hand_data = {
            "status": "hands_detected",
            "message": message,
            "hand_count": len(hands_info),
            "hands": hands_info
        }
        
        return output_frame, hand_data
    
    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Union[str, Dict]]:
        """
        Process frame to provide hand tracking information
        
        Args:
            frame (numpy.ndarray): Input frame to process
            
        Returns:
            tuple: (processed_frame, message_string)
                - processed_frame: Frame with visual indicators (optional)
                - message_string: Simple message for audio output
        """
        output_frame, hand_data = self._get_hand_data_internal(frame)
        # Return just the message string for clean audio output
        return output_frame, hand_data['message']
    
    def _determine_location(self, offset_x: float, offset_y: float) -> str:
        """
        Determine spatial location of hand in frame
        
        Args:
            offset_x: Horizontal offset from center (-1 to 1)
            offset_y: Vertical offset from center (-1 to 1)
            
        Returns:
            String describing location (e.g., "top-left", "center", "bottom-right")
        """
        # Define thresholds for center zone
        center_threshold = 0.3
        
        if abs(offset_x) < center_threshold and abs(offset_y) < center_threshold:
            return "center"
        
        # Determine vertical position
        if offset_y < -center_threshold:
            vertical = "top"
        elif offset_y > center_threshold:
            vertical = "bottom"
        else:
            vertical = "middle"
        
        # Determine horizontal position
        if offset_x < -center_threshold:
            horizontal = "left"
        elif offset_x > center_threshold:
            horizontal = "right"
        else:
            horizontal = "center"
        
        # Combine positions
        if vertical == "middle" and horizontal != "center":
            return horizontal
        elif horizontal == "center" and vertical != "middle":
            return vertical
        elif vertical != "middle" and horizontal != "center":
            return f"{vertical}-{horizontal}"
        else:
            return "center"
    
    def _determine_distance(self, size_ratio: float) -> Tuple[str, Dict]:
        """
        Determine relative distance based on hand size and generate beep parameters
        
        Args:
            size_ratio: Hand area as ratio of frame area
            
        Returns:
            Tuple of (distance_category, beep_params)
            - distance_category: String describing distance ("very close", "close", "medium", "far")
            - beep_params: Dictionary with frequency and interval for Geiger counter-style beeps
        """
        # Geiger counter style: closer = higher frequency beeps and shorter intervals
        if size_ratio > 0.25:
            # Very close - rapid high-pitched beeps
            return "very close", {"frequency": 1200, "interval": 0.1, "duration": 0.05}
        elif size_ratio > 0.12:
            # Close - moderate frequency and interval
            return "close", {"frequency": 900, "interval": 0.25, "duration": 0.08}
        elif size_ratio > 0.05:
            # Medium - slower beeps
            return "medium", {"frequency": 600, "interval": 0.5, "duration": 0.1}
        else:
            # Far - slow low-pitched beeps
            return "far", {"frequency": 400, "interval": 1.0, "duration": 0.12}
    
    def _generate_message(self, hands_info: List[Dict]) -> Union[str, Dict]:
        """
        Generate beep instructions for Geiger counter-style distance feedback
        
        Args:
            hands_info: List of hand information dictionaries
            
        Returns:
            Dictionary with beep instructions or string message if no hands detected
        """
        if not hands_info:
            return "No hands detected in view."
        
        if len(hands_info) == 1:
            hand = hands_info[0]
            # Return beep command with location description
            return {
                "type": "beep",
                "text": f"{hand['handedness']} hand at {hand['location']}",
                "beep_params": hand['beep_params']
            }
        else:
            # Multiple hands - use closest hand's beep params
            closest_hand = min(hands_info, key=lambda h: 1 - h['size_ratio'])  # Largest hand is closest
            hand_descriptions = [f"{h['handedness']} hand at {h['location']}" for h in hands_info]
            return {
                "type": "beep",
                "text": f"{len(hands_info)} hands: {', '.join(hand_descriptions)}",
                "beep_params": closest_hand['beep_params']
            }
    
    def get_hand_tracking_data(self, frame: np.ndarray) -> Dict:
        """
        Helper method for other processors to get full hand tracking data
        
        Args:
            frame: Input frame
            
        Returns:
            Dictionary with full hand tracking data (status, hands, locations, etc.)
        """
        # Get full hand data using internal method
        _, hand_data = self._get_hand_data_internal(frame)
        return hand_data
    
    def process_pointcloud(self, point_cloud_data: Dict) -> Tuple[Optional[Dict], Union[str, Dict]]:
        """
        Point cloud processing not implemented for hand tracking
        """
        return point_cloud_data, {"message": "HandTrackingProcessor does not process point clouds."}
    
    def __del__(self):
        """
        Clean up MediaPipe resources
        """
        if hasattr(self, 'hands'):
            self.hands.close()


processor = HandTrackingProcessor()
app = processor.app
