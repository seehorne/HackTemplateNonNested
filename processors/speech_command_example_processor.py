"""
Speech Command Example Processor

This is an example processor that demonstrates how to use the speech_input_processor
as a building block. It accepts voice commands to control image processing.

This serves as a template for building voice-controlled interactions.
"""

from .base_processor import BaseProcessor
from .speech_input_processor import SpeechInputProcessor
import numpy as np
import cv2
from typing import Dict, Union, Tuple, Optional

class SpeechCommandExampleProcessor(BaseProcessor):
    """
    Example processor demonstrating speech input building block usage.
    Accepts voice commands to apply simple image filters.
    """
    
    # Available filters
    FILTER_NONE = "none"
    FILTER_GRAYSCALE = "grayscale"
    FILTER_BLUR = "blur"
    FILTER_EDGE = "edge"
    FILTER_INVERT = "invert"
    
    def __init__(self):
        """Initialize the processor with speech input"""
        super().__init__()
        
        # Create speech input building block
        self.speech_input = SpeechInputProcessor(
            language="en-us",
            model_size="small"
        )
        
        # Current filter state
        self.current_filter = self.FILTER_NONE
        
        # Create voice command handler
        self.command_handler = self.speech_input.create_voice_command_handler({
            "original": lambda: self._set_filter(self.FILTER_NONE),
            "grayscale": lambda: self._set_filter(self.FILTER_GRAYSCALE),
            "gray": lambda: self._set_filter(self.FILTER_GRAYSCALE),
            "blur": lambda: self._set_filter(self.FILTER_BLUR),
            "edge": lambda: self._set_filter(self.FILTER_EDGE),
            "edges": lambda: self._set_filter(self.FILTER_EDGE),
            "invert": lambda: self._set_filter(self.FILTER_INVERT),
            "reset": lambda: self._set_filter(self.FILTER_NONE),
        })
    
    def _set_filter(self, filter_name: str):
        """Set the current filter"""
        self.current_filter = filter_name
    
    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Union[str, Dict]]:
        """
        Process frame by applying the current filter
        
        Args:
            frame (numpy.ndarray): Input frame to process
            
        Returns:
            tuple: (filtered_frame, status_message)
        """
        # Apply current filter
        output_frame = self._apply_filter(frame)
        
        # Return frame with current status
        return output_frame, {
            "message": f"Current filter: {self.current_filter}",
            "filter": self.current_filter,
            "available_commands": [
                "Say 'grayscale' or 'gray' to convert to grayscale",
                "Say 'blur' to apply blur filter",
                "Say 'edge' or 'edges' to detect edges",
                "Say 'invert' to invert colors",
                "Say 'original' or 'reset' to remove filters"
            ],
            "info": "Use speech input to control image filters. Send audio data to process_audio endpoint."
        }
    
    def _apply_filter(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply the current filter to the frame
        
        Args:
            frame: Input frame
            
        Returns:
            Filtered frame
        """
        if self.current_filter == self.FILTER_GRAYSCALE:
            # Convert to grayscale and back to BGR
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        elif self.current_filter == self.FILTER_BLUR:
            # Apply Gaussian blur
            return cv2.GaussianBlur(frame, (15, 15), 0)
        
        elif self.current_filter == self.FILTER_EDGE:
            # Edge detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        elif self.current_filter == self.FILTER_INVERT:
            # Invert colors
            return cv2.bitwise_not(frame)
        
        else:
            # No filter
            return frame.copy()
    
    def process_audio(self, audio_data: bytes) -> Dict:
        """
        Process audio input to recognize voice commands
        
        This method would typically be called from a custom endpoint
        or integrated into the main processing loop.
        
        Args:
            audio_data: Audio data as bytes (WAV format)
            
        Returns:
            Command execution result
        """
        # Use the command handler
        result = self.command_handler(audio_data)
        
        if result["executed"]:
            return {
                "success": True,
                "command": result["command"],
                "text": result["text"],
                "confidence": result["confidence"],
                "filter": self.current_filter,
                "message": f"Filter changed to: {self.current_filter}"
            }
        else:
            return {
                "success": False,
                "text": result.get("text", ""),
                "confidence": result.get("confidence", 0.0),
                "filter": self.current_filter,
                "message": result.get("message", "Command not recognized"),
                "error": result.get("error")
            }
    
    def process_pointcloud(self, point_cloud_data: Dict) -> Tuple[Optional[Dict], Union[str, Dict]]:
        """
        Point cloud processing not implemented for speech command example
        """
        return point_cloud_data, {
            "message": "SpeechCommandExampleProcessor does not process point clouds."
        }


# Create processor instance for the server
processor = SpeechCommandExampleProcessor()
app = processor.app
