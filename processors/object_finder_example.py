"""
Object Finder Example

This example demonstrates how to use the Object Finder Processor
to help blind users locate and reach objects with their hands.

This is a simple example showing the basic usage pattern.
"""

from .base_processor import BaseProcessor
from .object_finder_processor import ObjectFinderProcessor
import numpy as np
import cv2
from typing import Dict, Union, Tuple, Optional


class ObjectFinderExampleProcessor(BaseProcessor):
    """
    Example processor demonstrating Object Finder usage
    
    This processor:
    1. Uses ObjectFinderProcessor to guide users to objects
    2. Adds additional context and explanations
    3. Shows how to customize guidance messages
    """
    
    def __init__(self):
        """Initialize the example processor"""
        super().__init__()
        
        # Initialize the object finder
        self.object_finder = ObjectFinderProcessor(
            model_path="./models/yolo11n-seg.pt",
            guidance_threshold=50  # Pixels to consider "reached"
        )
        
        # State tracking
        self.first_use = True
        self.objects_reached = []
        
    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Union[str, Dict]]:
        """
        Process frame with enhanced guidance
        
        Args:
            frame: Input frame from camera
            
        Returns:
            tuple: (output_frame, enhanced_message)
        """
        # Get base guidance from object finder
        output_frame, message = self.object_finder.process_frame(frame)
        
        # Enhance the message for first-time users
        if self.first_use:
            if "Move hand" in message or "Almost there" in message:
                self.first_use = False
                message = "Welcome! I'll help you find objects. " + message
        
        # Track reached objects
        if "Object reached!" in message:
            # Extract object name (simple parsing)
            parts = message.split()
            if len(parts) >= 3:
                object_name = parts[2]  # "Object reached! Cup is..."
                if object_name not in self.objects_reached:
                    self.objects_reached.append(object_name)
                    message += f" You've found {len(self.objects_reached)} objects so far."
        
        # Add helpful tips based on state
        if "Show your hand" in message:
            message += " Tip: Hold your hand in front of the camera to start."
        
        if "No objects detected" in message:
            message += " Tip: Move the camera slowly to scan for objects."
        
        return output_frame, message
    
    def process_pointcloud(self, point_cloud_data: Dict) -> Tuple[Optional[Dict], Union[str, Dict]]:
        """Point cloud processing not implemented"""
        return point_cloud_data, {
            "message": "ObjectFinderExampleProcessor does not process point clouds."
        }


# Create processor instance for the server
processor = ObjectFinderExampleProcessor()
app = processor.app
