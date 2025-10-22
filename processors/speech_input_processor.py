"""
Speech Input Building Block Processor

This processor provides speech-to-text capabilities that can be used to allow
end users to provide input to processors using their voice (speech).

Designed as a building block for other processors requiring speech user input.
Examples: voice commands, dictation, verbal navigation, accessibility features.

CPU-Only: Uses Vosk for offline speech recognition, no GPU required.
"""

from .base_processor import BaseProcessor
import numpy as np
import json
import base64
from typing import Dict, Union, Tuple, Optional, List
import io
import struct
import wave


class SpeechInputProcessor(BaseProcessor):
    """
    Speech input building block processor for voice-based user input.
    Provides speech-to-text conversion using CPU-only offline recognition.
    """
    
    # Audio configuration
    SAMPLE_RATE = 16000  # Vosk works best with 16kHz
    CHANNELS = 1  # Mono
    SAMPLE_WIDTH = 2  # 16-bit
    
    # Recognition modes
    MODE_SINGLE_COMMAND = "single_command"
    MODE_CONTINUOUS = "continuous"
    MODE_DICTATION = "dictation"
    
    # Supported languages (can be extended)
    SUPPORTED_LANGUAGES = {
        "en-us": "English (US)",
        "en-gb": "English (UK)",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
        "ru": "Russian",
        "zh": "Chinese",
        "ja": "Japanese"
    }
    
    def __init__(self, language: str = "en-us", model_size: str = "small"):
        """
        Initialize Speech Input Processor
        
        Args:
            language: Language code (e.g., "en-us", "es", "fr")
            model_size: Model size - "small" (faster, ~50MB) or "large" (more accurate, ~1.5GB)
        """
        super().__init__()
        
        self.language = language
        self.model_size = model_size
        self.recognizer = None
        self.model = None
        
        # Lazy loading - model will be loaded on first use
        self._model_loaded = False
        self._load_error = None
    
    def _ensure_model_loaded(self):
        """
        Ensure Vosk model is loaded (lazy loading)
        """
        if self._model_loaded:
            return True
        
        if self._load_error:
            return False
        
        try:
            # Try to import vosk
            import vosk
            
            # Download/load model
            model_path = self._get_or_download_model()
            
            if not model_path:
                self._load_error = "Failed to get Vosk model"
                return False
            
            # Initialize model and recognizer
            self.model = vosk.Model(model_path)
            self.recognizer = vosk.KaldiRecognizer(self.model, self.SAMPLE_RATE)
            self.recognizer.SetWords(True)  # Enable word-level timestamps
            
            self._model_loaded = True
            return True
            
        except ImportError:
            self._load_error = "Vosk library not installed. Please install with: pip install vosk"
            return False
        except Exception as e:
            self._load_error = f"Error loading Vosk model: {str(e)}"
            return False
    
    def _get_or_download_model(self) -> Optional[str]:
        """
        Get path to Vosk model, downloading if necessary
        
        Returns:
            Path to model directory, or None if failed
        """
        import os
        
        # Model directory
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "vosk")
        os.makedirs(models_dir, exist_ok=True)
        
        # Model name based on language and size
        if self.model_size == "small":
            model_name = f"vosk-model-small-{self.language}"
        else:
            model_name = f"vosk-model-{self.language}"
        
        model_path = os.path.join(models_dir, model_name)
        
        # Check if model already exists
        if os.path.exists(model_path):
            return model_path
        
        # Model doesn't exist - provide instructions
        print(f"\n{'='*60}")
        print(f"Vosk model not found: {model_name}")
        print(f"{'='*60}")
        print(f"\nTo use speech recognition, please download a model:")
        print(f"\n1. Visit: https://alphacephei.com/vosk/models")
        print(f"2. Download: {model_name}.zip")
        print(f"3. Extract to: {models_dir}/")
        print(f"\nFor quick testing, you can use the lightweight model:")
        print(f"   wget https://alphacephei.com/vosk/models/{model_name}.zip")
        print(f"   unzip {model_name}.zip -d {models_dir}/")
        print(f"{'='*60}\n")
        
        return None
    
    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Union[str, Dict]]:
        """
        Process frame - provides demo information about the speech input processor
        
        Args:
            frame (numpy.ndarray): Input frame to process
            
        Returns:
            tuple: (original_frame, info_message)
        """
        # Check if model is loaded
        if not self._ensure_model_loaded():
            return frame, {
                "message": "Speech Input Processor - Model not loaded",
                "error": self._load_error,
                "instructions": "Please download a Vosk model to use speech recognition",
                "status": "not_ready"
            }
        
        return frame, {
            "message": "Speech Input Processor - Ready for audio input",
            "language": self.SUPPORTED_LANGUAGES.get(self.language, self.language),
            "sample_rate": self.SAMPLE_RATE,
            "status": "ready",
            "info": "This processor converts speech to text. Use recognize_speech() method with audio data.",
            "supported_modes": [
                self.MODE_SINGLE_COMMAND,
                self.MODE_CONTINUOUS,
                self.MODE_DICTATION
            ]
        }
    
    def process_pointcloud(self, point_cloud_data: Dict) -> Tuple[Optional[Dict], Union[str, Dict]]:
        """
        Point cloud processing not implemented for speech input
        """
        return point_cloud_data, {
            "message": "SpeechInputProcessor does not process point clouds."
        }
    
    def recognize_speech(
        self,
        audio_data: Union[bytes, np.ndarray],
        mode: str = MODE_SINGLE_COMMAND,
        partial_results: bool = False
    ) -> Dict:
        """
        Recognize speech from audio data
        
        Args:
            audio_data: Audio data as bytes (WAV format) or numpy array (PCM 16-bit)
            mode: Recognition mode (single_command, continuous, dictation)
            partial_results: Return partial results during recognition
            
        Returns:
            Dictionary with:
                - text: Recognized text
                - confidence: Confidence score (0.0 to 1.0)
                - words: List of word-level results with timestamps (if available)
                - partial: Whether this is a partial result
                - success: Whether recognition succeeded
        """
        if not self._ensure_model_loaded():
            return {
                "text": "",
                "confidence": 0.0,
                "words": [],
                "partial": False,
                "success": False,
                "error": self._load_error
            }
        
        try:
            # Convert audio data to PCM if needed
            pcm_data = self._prepare_audio_data(audio_data)
            
            # Reset recognizer for new recognition
            self.recognizer.Reset()
            
            # Process audio
            if self.recognizer.AcceptWaveform(pcm_data):
                # Final result
                result = json.loads(self.recognizer.Result())
                return self._format_result(result, partial=False)
            else:
                # Partial result
                if partial_results:
                    result = json.loads(self.recognizer.PartialResult())
                    return self._format_result(result, partial=True)
                else:
                    # No final result yet, return empty
                    return {
                        "text": "",
                        "confidence": 0.0,
                        "words": [],
                        "partial": False,
                        "success": False
                    }
                    
        except Exception as e:
            return {
                "text": "",
                "confidence": 0.0,
                "words": [],
                "partial": False,
                "success": False,
                "error": str(e)
            }
    
    def recognize_speech_streaming(
        self,
        audio_chunks: List[bytes],
        get_partial: bool = True
    ) -> List[Dict]:
        """
        Recognize speech from streaming audio chunks
        
        Args:
            audio_chunks: List of audio data chunks (bytes)
            get_partial: Whether to include partial results
            
        Returns:
            List of recognition results (one per chunk or final result)
        """
        if not self._ensure_model_loaded():
            return [{
                "text": "",
                "confidence": 0.0,
                "success": False,
                "error": self._load_error
            }]
        
        results = []
        self.recognizer.Reset()
        
        try:
            for chunk in audio_chunks:
                pcm_data = self._prepare_audio_data(chunk)
                
                if self.recognizer.AcceptWaveform(pcm_data):
                    # Final result for this chunk
                    result = json.loads(self.recognizer.Result())
                    results.append(self._format_result(result, partial=False))
                elif get_partial:
                    # Partial result
                    result = json.loads(self.recognizer.PartialResult())
                    results.append(self._format_result(result, partial=True))
            
            # Get final result
            final_result = json.loads(self.recognizer.FinalResult())
            if final_result.get("text"):
                results.append(self._format_result(final_result, partial=False))
            
            return results if results else [{
                "text": "",
                "confidence": 0.0,
                "success": False
            }]
            
        except Exception as e:
            return [{
                "text": "",
                "confidence": 0.0,
                "success": False,
                "error": str(e)
            }]
    
    def _prepare_audio_data(self, audio_data: Union[bytes, np.ndarray]) -> bytes:
        """
        Prepare audio data for recognition (convert to PCM 16-bit mono)
        
        Args:
            audio_data: Audio data as bytes or numpy array
            
        Returns:
            PCM audio data as bytes
        """
        if isinstance(audio_data, np.ndarray):
            # Convert numpy array to bytes
            if audio_data.dtype != np.int16:
                # Convert to int16
                audio_data = (audio_data * 32767).astype(np.int16)
            return audio_data.tobytes()
        
        elif isinstance(audio_data, bytes):
            # Check if it's WAV format
            if audio_data[:4] == b'RIFF':
                # Extract PCM data from WAV
                return self._extract_pcm_from_wav(audio_data)
            else:
                # Assume it's already PCM
                return audio_data
        
        else:
            raise ValueError(f"Unsupported audio data type: {type(audio_data)}")
    
    def _extract_pcm_from_wav(self, wav_data: bytes) -> bytes:
        """
        Extract PCM data from WAV file bytes
        
        Args:
            wav_data: WAV file as bytes
            
        Returns:
            PCM audio data
        """
        wav_buffer = io.BytesIO(wav_data)
        with wave.open(wav_buffer, 'rb') as wav_file:
            # Check format
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            framerate = wav_file.getframerate()
            
            # Read all frames
            pcm_data = wav_file.readframes(wav_file.getnframes())
            
            # Convert if needed (Vosk expects mono 16kHz 16-bit)
            if channels != self.CHANNELS or framerate != self.SAMPLE_RATE or sample_width != self.SAMPLE_WIDTH:
                # For now, we'll just warn and use as-is
                # In production, you'd want to resample
                print(f"Warning: Audio format mismatch. Expected {self.CHANNELS}ch, {self.SAMPLE_RATE}Hz, {self.SAMPLE_WIDTH*8}bit")
                print(f"Got {channels}ch, {framerate}Hz, {sample_width*8}bit")
            
            return pcm_data
    
    def _format_result(self, result: Dict, partial: bool = False) -> Dict:
        """
        Format recognition result to standard format
        
        Args:
            result: Raw result from Vosk
            partial: Whether this is a partial result
            
        Returns:
            Formatted result dictionary
        """
        text = result.get("text", result.get("partial", ""))
        
        # Calculate confidence from word results if available
        confidence = 0.0
        words = []
        
        if "result" in result:
            # Word-level results available
            word_results = result["result"]
            if word_results:
                # Average confidence
                confidences = [w.get("conf", 0.0) for w in word_results]
                confidence = sum(confidences) / len(confidences) if confidences else 0.0
                
                # Extract word information
                words = [
                    {
                        "word": w.get("word", ""),
                        "start": w.get("start", 0.0),
                        "end": w.get("end", 0.0),
                        "conf": w.get("conf", 0.0)
                    }
                    for w in word_results
                ]
        else:
            # No word-level info, estimate confidence
            confidence = 0.8 if text else 0.0
        
        return {
            "text": text,
            "confidence": confidence,
            "words": words,
            "partial": partial,
            "success": bool(text)
        }
    
    def get_speech_from_base64_wav(self, wav_base64: str) -> Dict:
        """
        Helper: Recognize speech from base64-encoded WAV data
        
        Args:
            wav_base64: Base64-encoded WAV audio
            
        Returns:
            Recognition result dictionary
        """
        try:
            wav_data = base64.b64decode(wav_base64)
            return self.recognize_speech(wav_data)
        except Exception as e:
            return {
                "text": "",
                "confidence": 0.0,
                "success": False,
                "error": f"Error decoding base64 audio: {str(e)}"
            }
    
    def create_voice_command_handler(self, commands: Dict[str, callable]) -> callable:
        """
        Helper: Create a voice command handler function
        
        Args:
            commands: Dictionary mapping command phrases to callback functions
                     Example: {"start": lambda: print("Starting"), "stop": lambda: print("Stopping")}
        
        Returns:
            Handler function that takes audio data and executes matching commands
        """
        def handler(audio_data: Union[bytes, np.ndarray]) -> Dict:
            """
            Process audio and execute matching command
            
            Returns:
                Dictionary with command execution result
            """
            result = self.recognize_speech(audio_data)
            
            if not result["success"]:
                return {
                    "command": None,
                    "executed": False,
                    "error": result.get("error", "Recognition failed")
                }
            
            text = result["text"].lower().strip()
            
            # Check for matching commands
            for cmd_phrase, callback in commands.items():
                if cmd_phrase.lower() in text:
                    try:
                        callback()
                        return {
                            "command": cmd_phrase,
                            "text": text,
                            "confidence": result["confidence"],
                            "executed": True
                        }
                    except Exception as e:
                        return {
                            "command": cmd_phrase,
                            "text": text,
                            "executed": False,
                            "error": str(e)
                        }
            
            # No matching command
            return {
                "command": None,
                "text": text,
                "confidence": result["confidence"],
                "executed": False,
                "message": "No matching command found"
            }
        
        return handler


# Create processor instance for the server
processor = SpeechInputProcessor()
app = processor.app
