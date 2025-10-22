"""
Speech Input Building Block Module

This module provides utilities for other processors to easily integrate
speech-to-text input capabilities. It can be imported and used as a
building block in any processor that needs speech input.

Example usage in another processor:

    from processors.speech_input_building_block import SpeechInputHelper
    
    class MyProcessor(BaseProcessor):
        def __init__(self):
            super().__init__()
            self.speech_helper = SpeechInputHelper()
        
        def process_with_speech(self, frame, audio_data):
            # Get transcription
            text = self.speech_helper.transcribe(audio_data)
            # Process frame with text
            result = self.do_something(frame, text)
            return result
"""

import numpy as np
import logging
from typing import Union, Optional

logger = logging.getLogger(__name__)

class SpeechInputHelper:
    """
    Helper class to provide speech-to-text capabilities to any processor.
    
    This is a lightweight wrapper around the Whisper model that can be
    instantiated in any processor to add speech input functionality.
    """
    
    def __init__(self, model_name: str = "distil-whisper/distil-small.en", device: str = "cpu"):
        """
        Initialize the speech input helper.
        
        Args:
            model_name: HuggingFace model name for Whisper
            device: Device to run on (default: "cpu")
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.sample_rate = 16000
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model lazily"""
        try:
            from transformers import pipeline
            logger.info(f"Loading speech recognition model: {self.model_name}")
            self.model = pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
                device=self.device
            )
            logger.info("Speech recognition model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load speech recognition model: {e}")
            logger.warning("Speech input will not be available")
    
    def is_available(self) -> bool:
        """Check if speech recognition is available"""
        return self.model is not None
    
    def transcribe(self, audio_data: Union[bytes, np.ndarray]) -> str:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data: Audio data as bytes or numpy array
            
        Returns:
            Transcribed text string
        """
        if not self.is_available():
            logger.error("Speech recognition model not available")
            return ""
        
        try:
            # Convert bytes to numpy array if needed
            if isinstance(audio_data, bytes):
                audio_array = np.frombuffer(audio_data, dtype=np.float32)
            else:
                audio_array = audio_data
            
            # Ensure proper format
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32) / 32768.0
            
            # Run inference
            result = self.model(audio_array, return_timestamps=False)
            text = result.get("text", "").strip()
            
            logger.info(f"Transcription: {text}")
            return text
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""
    
    def transcribe_with_timestamps(self, audio_data: Union[bytes, np.ndarray]) -> dict:
        """
        Transcribe audio with word-level timestamps.
        
        Args:
            audio_data: Audio data as bytes or numpy array
            
        Returns:
            Dictionary with text and timestamps
        """
        if not self.is_available():
            logger.error("Speech recognition model not available")
            return {"text": "", "chunks": []}
        
        try:
            # Convert bytes to numpy array if needed
            if isinstance(audio_data, bytes):
                audio_array = np.frombuffer(audio_data, dtype=np.float32)
            else:
                audio_array = audio_data
            
            # Ensure proper format
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32) / 32768.0
            
            # Run inference with timestamps
            result = self.model(audio_array, return_timestamps=True)
            
            return {
                "text": result.get("text", "").strip(),
                "chunks": result.get("chunks", [])
            }
            
        except Exception as e:
            logger.error(f"Transcription with timestamps error: {e}")
            return {"text": "", "chunks": []}


class VoiceCommandProcessor:
    """
    Helper class for processing voice commands in a processor.
    
    This can be used to parse transcribed text for specific commands
    or intents, making it easier to create voice-controlled processors.
    """
    
    def __init__(self, commands: Optional[dict] = None):
        """
        Initialize voice command processor.
        
        Args:
            commands: Dictionary mapping command keywords to actions
                     Example: {"start": self.start_action, "stop": self.stop_action}
        """
        self.commands = commands or {}
    
    def add_command(self, keyword: str, action):
        """
        Add a voice command.
        
        Args:
            keyword: Trigger word/phrase
            action: Function to call when keyword is detected
        """
        self.commands[keyword.lower()] = action
    
    def process_text(self, text: str) -> Optional[str]:
        """
        Process transcribed text for commands.
        
        Args:
            text: Transcribed text from speech
            
        Returns:
            Detected command keyword or None
        """
        text_lower = text.lower()
        
        for keyword in self.commands.keys():
            if keyword in text_lower:
                logger.info(f"Detected command: {keyword}")
                return keyword
        
        return None
    
    def execute_command(self, text: str, *args, **kwargs):
        """
        Detect and execute a command from transcribed text.
        
        Args:
            text: Transcribed text
            *args, **kwargs: Arguments to pass to the command action
            
        Returns:
            Result of the command action or None if no command detected
        """
        command = self.process_text(text)
        
        if command and command in self.commands:
            action = self.commands[command]
            try:
                return action(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error executing command '{command}': {e}")
                return None
        
        return None


# Example usage documentation
__doc__ += """

## Building Block Example

Here's how to use this module in your processor:

```python
from processors.speech_input_building_block import SpeechInputHelper, VoiceCommandProcessor
from processors.base_processor import BaseProcessor

class MyVoiceControlledProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        
        # Add speech input capability
        self.speech = SpeechInputHelper()
        
        # Optional: Add voice command handling
        self.voice_commands = VoiceCommandProcessor()
        self.voice_commands.add_command("zoom in", self.zoom_in)
        self.voice_commands.add_command("zoom out", self.zoom_out)
    
    def process_frame_with_audio(self, frame, audio_data):
        # Transcribe audio
        text = self.speech.transcribe(audio_data)
        
        # Check for voice commands
        self.voice_commands.execute_command(text)
        
        # Use transcribed text in processing
        result = f"You said: {text}"
        return frame, result
```

## Direct Usage

You can also use the speech processor directly via HTTP:

```python
import requests
import base64

# Send audio to speech processor
response = requests.post(
    "http://127.0.0.1:8014/transcribe",
    json={
        "audio": base64.b64encode(audio_bytes).decode(),
        "format": "raw"
    }
)

transcription = response.json()["transcription"]
```
"""
