"""
Example Photo Capture Processor - Demonstrates Camera Aiming Building Block Usage

This is an example processor that shows how to integrate the Camera Aiming Processor
as a building block. It guides the user to center a person in the frame, then
simulates capturing a photo when properly aligned.

This serves as a template for other processors that need camera aiming functionality.
"""

from .base_processor import BaseProcessor
from .camera_aiming_processor import CameraAimingProcessor
import numpy as np
import cv2
from typing import Dict, Union, Tuple, List, Optional
import time

class PhotoCaptureExampleProcessor(BaseProcessor):
    """
    Example processor demonstrating camera aiming building block integration.
    
    This processor:
    1. Uses camera aiming to help user center a person
    2. Provides countdown when properly aligned
    3. Simulates photo capture
    4. Can be used as a template for similar processors
    """
    
    def __init__(self, target_class="person", countdown_seconds=3):
        """
        Initialize Photo Capture Example Processor
        
        Args:
            target_class (str): Object class to aim at (default: "person")
            countdown_seconds (int): Seconds to count down before capture
        """
        super().__init__()
        
        # Initialize camera aiming building block
        self.aiming_helper = CameraAimingProcessor(target_class=target_class)
        
        # State management
        self.countdown_seconds = countdown_seconds
        self.capture_ready = False
        self.countdown_start_time = None
        self.captured = False
        self.capture_cooldown = 5  # Seconds before allowing another capture
        self.last_capture_time = None
        
    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Union[str, Dict]]:
        """
        Process frame with camera aiming and photo capture logic
        
        Args:
            frame (numpy.ndarray): Input frame to process
            
        Returns:
            tuple: (processed_frame, guidance_or_result)
        """
        current_time = time.time()
        
        # Check if we're in cooldown after a capture
        if self.last_capture_time and (current_time - self.last_capture_time) < self.capture_cooldown:
            remaining = int(self.capture_cooldown - (current_time - self.last_capture_time))
            message = f"Photo captured! Wait {remaining} seconds before next capture."
            
            # Draw simple "captured" indicator
            output = self._draw_capture_indicator(frame.copy())
            return output, message
        
        # Reset capture state after cooldown
        if self.last_capture_time and (current_time - self.last_capture_time) >= self.capture_cooldown:
            self.captured = False
            self.capture_ready = False
            self.countdown_start_time = None
            self.last_capture_time = None
        
        # Get aiming guidance from the building block
        guidance = self.aiming_helper.get_aiming_guidance(frame)
        
        # Check if object is perfectly aligned
        if guidance['status'] == 'perfect':
            # Start countdown if not already started
            if not self.countdown_start_time:
                self.countdown_start_time = current_time
                self.capture_ready = True
            
            # Calculate countdown
            elapsed = current_time - self.countdown_start_time
            remaining = max(0, self.countdown_seconds - elapsed)
            
            if remaining > 0:
                # Still counting down
                message = f"{guidance['object_name']} perfectly centered! Capturing in {int(remaining + 1)}..."
                output = self._draw_countdown(frame.copy(), int(remaining + 1))
                return output, message
            else:
                # Time to capture!
                self.captured = True
                self.last_capture_time = current_time
                output = self._draw_capture_flash(frame.copy())
                
                result = {
                    "status": "captured",
                    "message": "Photo captured successfully!",
                    "object_name": guidance['object_name'],
                    "confidence": guidance['confidence'],
                    "timestamp": current_time
                }
                return output, result
        
        else:
            # Not perfectly aligned, reset countdown
            self.countdown_start_time = None
            self.capture_ready = False
            
            # Return the aiming guidance
            # Get visual frame from aiming processor
            output, _ = self.aiming_helper.process_frame(frame)
            return output, guidance['message']
    
    def _draw_countdown(self, frame: np.ndarray, seconds: int) -> np.ndarray:
        """Draw countdown number on frame"""
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        
        # Draw large countdown number
        font = cv2.FONT_HERSHEY_BOLD
        text = str(seconds)
        font_scale = 8
        thickness = 15
        
        # Get text size for centering
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = center[0] - text_w // 2
        text_y = center[1] + text_h // 2
        
        # Draw text with background
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)
        
        return frame
    
    def _draw_capture_flash(self, frame: np.ndarray) -> np.ndarray:
        """Draw a flash effect for capture"""
        # Add white overlay for flash effect
        flash = np.ones_like(frame) * 255
        output = cv2.addWeighted(frame, 0.3, flash.astype(np.uint8), 0.7, 0)
        
        # Add "CAPTURED" text
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_BOLD
        text = "CAPTURED!"
        font_scale = 3
        thickness = 5
        
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = (w - text_w) // 2
        text_y = (h + text_h) // 2
        
        cv2.putText(output, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)
        
        return output
    
    def _draw_capture_indicator(self, frame: np.ndarray) -> np.ndarray:
        """Draw indicator that photo was captured"""
        h, w = frame.shape[:2]
        
        # Draw checkmark in corner
        cv2.circle(frame, (w - 50, 50), 30, (0, 255, 0), -1)
        cv2.putText(frame, "✓", (w - 65, 65), cv2.FONT_HERSHEY_BOLD, 2, (255, 255, 255), 3)
        
        return frame
    
    def process_pointcloud(self, point_cloud_data: Dict) -> Tuple[Optional[Dict], Union[str, Dict]]:
        """
        Point cloud processing not implemented for this example
        """
        return point_cloud_data, {"message": "PhotoCaptureExampleProcessor does not process point clouds."}


# Note: This is an example processor for demonstration purposes.
# To enable it, add it to processor_config.json with a unique ID and port.
# For this example, we don't instantiate it as the default processor.

# Example configuration for processor_config.json:
# {
#   "14": {
#     "host": "127.0.0.1",
#     "port": 8015,
#     "name": "photo_capture_example_processor",
#     "conda_env": "whatsai",
#     "dependencies": [],
#     "expects_input": "image",
#     "description": "Example processor demonstrating camera aiming building block integration for photo capture.",
#     "enabled": false
#   }
# }
