from .base_processor import BaseProcessor
from typing import Dict, Union, Tuple, Optional
import numpy as np
import base64
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeechInputProcessor(BaseProcessor):
    """
    Building block processor for speech input recognition.
    
    This processor can be used by other processors to accept speech input
    and convert it to text. It uses Whisper (via transformers) for CPU-based
    speech recognition.
    
    Expected input: Audio data as base64 encoded string or raw audio frames
    Output: Transcribed text
    """
    
    def __init__(self):
        """Initialize the speech input processor with Whisper model"""
        super().__init__()
        self.model = None
        self.processor_obj = None
        self.sample_rate = 16000  # Whisper expects 16kHz audio
        
        try:
            from transformers import pipeline
            logger.info("Loading Whisper model for speech recognition...")
            # Use distil-whisper for faster CPU inference
            self.model = pipeline(
                "automatic-speech-recognition",
                model="distil-whisper/distil-small.en",
                device="cpu"
            )
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            logger.info("Speech recognition will not be available")
    
    def process_audio_input(self, audio_data: Union[bytes, np.ndarray]) -> str:
        """
        Process audio input and return transcribed text.
        
        Args:
            audio_data: Either bytes (raw audio) or numpy array
            
        Returns:
            Transcribed text string
        """
        if self.model is None:
            return "Error: Speech recognition model not loaded"
        
        try:
            # If audio_data is bytes, convert to numpy array
            if isinstance(audio_data, bytes):
                audio_array = np.frombuffer(audio_data, dtype=np.float32)
            else:
                audio_array = audio_data
            
            # Ensure audio is in the right format (float32, mono)
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32) / 32768.0
            
            # Run speech recognition
            result = self.model(audio_array, return_timestamps=False)
            text = result.get("text", "")
            
            logger.info(f"Transcribed: {text}")
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return f"Error: {str(e)}"
    
    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Union[str, Dict]]:
        """
        Process image frame - this processor doesn't process images directly.
        Instead, it's designed to be used as a building block for other processors.
        
        Args:
            frame: Input image frame
            
        Returns:
            Original frame and informational message
        """
        result_payload = {
            "message": "Speech Input Processor - Building Block",
            "description": "This processor is designed to accept audio input. Use it as a building block in other processors.",
            "capabilities": [
                "Speech-to-text transcription",
                "CPU-only inference",
                "Supports 16kHz audio input",
                "Based on Whisper model"
            ],
            "usage": "Send audio data via custom endpoint or integrate with other processors"
        }
        
        return frame, result_payload
    
    def process_pointcloud(self, point_cloud_data: Dict) -> Tuple[Optional[Dict], Union[str, Dict]]:
        """
        Point cloud processing not supported by this processor.
        
        Args:
            point_cloud_data: Input point cloud data
            
        Returns:
            Original point cloud and informational message
        """
        return point_cloud_data, {
            "message": "Speech Input Processor does not process point cloud data",
            "capabilities": ["speech-to-text only"]
        }


# Create processor instance
processor = SpeechInputProcessor()
app = processor.app

# Add custom endpoint for audio processing
from fastapi import HTTPException
from pydantic import BaseModel

class AudioRequest(BaseModel):
    audio: str  # Base64 encoded audio data
    format: Optional[str] = "raw"  # raw, wav, etc.

@app.post("/transcribe")
async def transcribe_audio(request: AudioRequest):
    """
    Custom endpoint for transcribing audio.
    
    Expected format:
    - audio: base64 encoded audio data
    - format: "raw" (default) or "wav"
    """
    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(request.audio)
        
        # Process audio
        transcription = processor.process_audio_input(audio_bytes)
        
        return {
            "transcription": transcription,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error in transcribe endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": processor.model is not None,
        "processor_type": "speech_input"
    }
