"""
Hand Guidance Example Processor

This is an example processor that demonstrates how to use the hand_tracking_processor
as a building block. It guides users to move their hand to the center of the frame
using non-visual audio cues.

This serves as a template for building more complex hand-based interactions.
"""

from .base_processor import BaseProcessor
from .hand_tracking_processor import HandTrackingProcessor
import numpy as np
from typing import Dict, Union, Tuple, Optional

class HandGuidanceExampleProcessor(BaseProcessor):
    """
    Example processor demonstrating hand tracking building block usage.
    Guides user to center their hand in the frame.
    """
    
    def __init__(self):
        """Initialize the processor with a hand tracker"""
        super().__init__()
        
        # Create hand tracking building block
        self.hand_tracker = HandTrackingProcessor(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=1  # Only track one hand for simplicity
        )
    
    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Union[str, Dict]]:
        """
        Process frame to guide user to center their hand
        
        Args:
            frame (numpy.ndarray): Input frame to process
            
        Returns:
            tuple: (processed_frame, guidance_message)
        """
        # Get full hand tracking data from building block
        hand_data = self.hand_tracker.get_hand_tracking_data(frame)
        
        # Check if hand is detected
        if hand_data['status'] == 'no_hands':
            return frame, "Please show your hand in the camera view."
        
        # Get the first hand
        hand = hand_data['hands'][0]
        
        # Generate guidance based on hand position
        message = self._generate_guidance(hand)
        
        # Return the processed frame (with landmarks drawn by hand tracker)
        # Note: We could also draw our own custom overlays here
        output_frame, _ = self.hand_tracker._get_hand_data_internal(frame)
        
        return output_frame, message
    
    def _generate_guidance(self, hand: Dict) -> str:
        """
        Generate guidance message based on hand position
        
        Args:
            hand: Hand information dictionary from hand tracker
            
        Returns:
            Guidance message string
        """
        location = hand['location']
        distance = hand['distance']
        handedness = hand['handedness']
        
        # Check if hand is perfectly centered
        if location == 'center' and distance == 'medium':
            return f"Perfect! {handedness} hand is centered at ideal distance. Well done!"
        
        # Check if centered but wrong distance
        if location == 'center':
            if distance == 'very close':
                return f"{handedness} hand centered but too close. Move camera back."
            elif distance == 'close':
                return f"{handedness} hand centered. Move slightly back for optimal distance."
            elif distance == 'far':
                return f"{handedness} hand centered but too far. Move camera closer."
        
        # Not centered - provide directional guidance
        directions = []
        
        # Parse location string
        if 'top' in location:
            directions.append("down")
        elif 'bottom' in location:
            directions.append("up")
        
        if 'left' in location:
            directions.append("right")
        elif 'right' in location:
            directions.append("left")
        elif location in ['left', 'right']:
            # Handle pure left/right
            if location == 'left':
                directions.append("right")
            else:
                directions.append("left")
        
        # Build message
        if directions:
            direction_text = " and ".join(directions)
            message = f"Move your {handedness.lower()} hand {direction_text} to center it."
        else:
            message = f"{handedness} hand detected at {location}."
        
        # Add distance guidance if needed
        if distance == 'very close':
            message += " Also move camera back."
        elif distance == 'far':
            message += " Also move camera closer."
        
        return message
    
    def process_pointcloud(self, point_cloud_data: Dict) -> Tuple[Optional[Dict], Union[str, Dict]]:
        """
        Point cloud processing not implemented for hand guidance
        """
        return point_cloud_data, {"message": "HandGuidanceExampleProcessor does not process point clouds."}


# Create processor instance for the server
processor = HandGuidanceExampleProcessor()
app = processor.app
