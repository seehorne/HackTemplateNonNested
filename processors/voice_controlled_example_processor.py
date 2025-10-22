"""
Example Processor Using Speech Input Building Block

This is a demonstration processor that shows how to use the speech input
building block in a custom processor. It combines image processing with
voice commands.

This processor:
1. Processes images
2. Accepts voice commands to control processing
3. Uses speech input as a building block
"""

from .base_processor import BaseProcessor
from .speech_input_building_block import SpeechInputHelper, VoiceCommandProcessor
from typing import Dict, Union, Tuple, Optional
import numpy as np
import cv2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoiceControlledImageProcessor(BaseProcessor):
    """
    Example processor that demonstrates speech input building block usage.
    
    This processor can:
    - Process images with various filters
    - Accept voice commands to change filters
    - Demonstrate how to integrate SpeechInputHelper
    """
    
    def __init__(self):
        """Initialize the voice-controlled image processor"""
        super().__init__()
        
        # Initialize speech input helper (building block)
        self.speech = SpeechInputHelper()
        
        # Set up voice commands
        self.voice_commands = VoiceCommandProcessor({
            "grayscale": "grayscale",
            "gray scale": "grayscale",
            "blur": "blur",
            "edge": "edges",
            "edges": "edges",
            "normal": "normal",
            "reset": "normal",
            "original": "normal"
        })
        
        # Current processing mode
        self.current_mode = "normal"
        
        logger.info("Voice-controlled image processor initialized")
        logger.info("Available commands: grayscale, blur, edges, normal/reset")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Union[str, Dict]]:
        """
        Process the image frame based on current mode.
        
        Args:
            frame: Input image frame
            
        Returns:
            Processed frame and result payload
        """
        try:
            # Apply the current filter
            if self.current_mode == "grayscale":
                processed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
                mode_desc = "Grayscale filter applied"
                
            elif self.current_mode == "blur":
                processed = cv2.GaussianBlur(frame, (15, 15), 0)
                mode_desc = "Gaussian blur applied"
                
            elif self.current_mode == "edges":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                processed = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                mode_desc = "Edge detection applied"
                
            else:  # normal
                processed = frame.copy()
                mode_desc = "Original image"
            
            result_payload = {
                "mode": self.current_mode,
                "description": mode_desc,
                "speech_available": self.speech.is_available(),
                "voice_commands": list(set(self.voice_commands.commands.values())),
                "usage": "Say a command like 'grayscale', 'blur', 'edges', or 'normal' to change filters"
            }
            
            return processed, result_payload
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame, {"error": str(e)}
    
    def process_with_voice_command(self, frame: np.ndarray, audio_data: Union[bytes, np.ndarray]) -> Tuple[Optional[np.ndarray], Union[str, Dict]]:
        """
        Process frame with voice command input.
        
        This demonstrates how to combine image processing with speech input.
        
        Args:
            frame: Input image frame
            audio_data: Audio data containing voice command
            
        Returns:
            Processed frame and result payload
        """
        try:
            # Transcribe the audio using the speech building block
            transcription = self.speech.transcribe(audio_data)
            
            # Check for voice commands
            command = self.voice_commands.process_text(transcription)
            
            if command:
                # Map command to mode
                self.current_mode = self.voice_commands.commands.get(command.lower(), "normal")
                logger.info(f"Voice command detected: '{transcription}' -> mode: {self.current_mode}")
                command_detected = True
            else:
                logger.info(f"No command detected in: '{transcription}'")
                command_detected = False
            
            # Process the frame with the current mode
            processed_frame, result = self.process_frame(frame)
            
            # Add voice command info to result
            if isinstance(result, dict):
                result["transcription"] = transcription
                result["command_detected"] = command_detected
            
            return processed_frame, result
            
        except Exception as e:
            logger.error(f"Error processing with voice command: {e}")
            return frame, {"error": str(e), "transcription": ""}
    
    def process_pointcloud(self, point_cloud_data: Dict) -> Tuple[Optional[Dict], Union[str, Dict]]:
        """Point cloud processing not supported"""
        return point_cloud_data, {
            "message": "Voice-controlled image processor does not process point clouds",
            "processor_type": "image+voice"
        }


# Create processor instance
processor = VoiceControlledImageProcessor()
app = processor.app

# Add custom endpoint for voice-controlled processing
from fastapi import HTTPException
from pydantic import BaseModel
import base64

class VoiceProcessRequest(BaseModel):
    image: str  # Base64 encoded image
    audio: Optional[str] = None  # Base64 encoded audio (optional)

@app.post("/process_with_voice")
async def process_with_voice(request: VoiceProcessRequest):
    """
    Process image with optional voice command.
    
    If audio is provided, it will be transcribed and checked for commands.
    The image will then be processed according to the current mode.
    """
    try:
        # Decode image
        if request.image.startswith('data:image/jpeg;base64,'):
            encoded_data = request.image.split('base64,')[1]
        else:
            encoded_data = request.image
        
        decoded_data = base64.b64decode(encoded_data)
        nparr = np.frombuffer(decoded_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise ValueError("Failed to decode image")
        
        # Process with or without audio
        if request.audio:
            # Decode audio
            audio_bytes = base64.b64decode(request.audio)
            processed_frame, result = processor.process_with_voice_command(frame, audio_bytes)
        else:
            processed_frame, result = processor.process_frame(frame)
        
        # Encode processed frame
        response_dict = {"result": result}
        if isinstance(processed_frame, np.ndarray):
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
            success, buffer = cv2.imencode('.jpg', processed_frame, encode_param)
            if success:
                processed_image_b64 = base64.b64encode(buffer).decode('utf-8')
                response_dict["image"] = f"data:image/jpeg;base64,{processed_image_b64}"
        
        return response_dict
        
    except Exception as e:
        logger.error(f"Error in process_with_voice endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/set_mode")
async def set_mode(mode: str):
    """
    Manually set the processing mode.
    
    Args:
        mode: One of "normal", "grayscale", "blur", "edges"
    """
    valid_modes = ["normal", "grayscale", "blur", "edges"]
    
    if mode not in valid_modes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode. Must be one of: {', '.join(valid_modes)}"
        )
    
    processor.current_mode = mode
    return {
        "status": "success",
        "mode": processor.current_mode,
        "message": f"Processing mode set to {mode}"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "speech_available": processor.speech.is_available(),
        "current_mode": processor.current_mode,
        "processor_type": "voice_controlled_image"
    }
