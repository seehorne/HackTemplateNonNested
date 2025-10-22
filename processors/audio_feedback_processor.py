"""
Audio Feedback Building Block Processor

This processor provides non-verbal audio feedback that can be used to render:
- Beeps (geiger counter style)
- Tones with varying pitch and frequency
- Variable intensity/volume feedback
- Different audio patterns for different states

Designed as a building block for other processors requiring audio feedback.
Examples: proximity sensors, alignment cues, status indicators, scanning feedback.

CPU-Only: Uses numpy for audio generation, no GPU required.
"""

from .base_processor import BaseProcessor
import numpy as np
import base64
import cv2
from typing import Dict, Union, Tuple, Optional, List
import struct
import io


class AudioFeedbackProcessor(BaseProcessor):
    """
    Audio feedback building block processor for non-verbal audio cues.
    Generates various audio patterns for accessibility and feedback.
    """
    
    # Audio configuration
    SAMPLE_RATE = 44100  # Standard audio sample rate (Hz)
    
    # Audio types
    AUDIO_TYPE_BEEP = "beep"
    AUDIO_TYPE_TONE = "tone"
    AUDIO_TYPE_SWEEP = "sweep"
    AUDIO_TYPE_PULSE = "pulse"
    AUDIO_TYPE_GEIGER = "geiger"
    
    # Presets for common use cases
    PRESET_SCANNING = "scanning"
    PRESET_PROXIMITY = "proximity"
    PRESET_ALIGNMENT = "alignment"
    PRESET_SUCCESS = "success"
    PRESET_WARNING = "warning"
    PRESET_ERROR = "error"
    
    def __init__(self):
        """Initialize Audio Feedback Processor"""
        super().__init__()
        
    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Union[str, Dict]]:
        """
        Process frame - generates a demo beep to demonstrate the audio feedback
        
        Args:
            frame (numpy.ndarray): Input frame to process
            
        Returns:
            tuple: (original_frame, result_with_audio)
        """
        # Generate a demo beep to show that the processor is working
        audio_data = self.generate_audio_feedback(
            audio_type=self.AUDIO_TYPE_BEEP,
            frequency=440,
            duration=0.2,
            intensity=0.5
        )
        
        return frame, {
            "message": "Audio Feedback Processor - Demo beep generated",
            "audio": audio_data,
            "info": "This processor generates non-verbal audio feedback. Use generate_audio_feedback() method for custom audio.",
            "available_types": [
                self.AUDIO_TYPE_BEEP,
                self.AUDIO_TYPE_TONE,
                self.AUDIO_TYPE_SWEEP,
                self.AUDIO_TYPE_PULSE,
                self.AUDIO_TYPE_GEIGER
            ],
            "available_presets": [
                self.PRESET_SCANNING,
                self.PRESET_PROXIMITY,
                self.PRESET_ALIGNMENT,
                self.PRESET_SUCCESS,
                self.PRESET_WARNING,
                self.PRESET_ERROR
            ]
        }
    
    def process_pointcloud(self, point_cloud_data: Dict) -> Tuple[Optional[Dict], Union[str, Dict]]:
        """
        Point cloud processing not implemented for audio feedback
        """
        return point_cloud_data, {
            "message": "AudioFeedbackProcessor does not process point clouds."
        }
    
    def generate_audio_feedback(
        self,
        audio_type: str = AUDIO_TYPE_BEEP,
        duration: float = 0.2,
        frequency: float = 440.0,
        intensity: float = 0.5,
        pattern: Optional[List[float]] = None,
        preset: Optional[str] = None
    ) -> Dict:
        """
        Generate audio feedback based on parameters
        
        Args:
            audio_type (str): Type of audio (beep, tone, sweep, pulse, geiger)
            duration (float): Duration in seconds
            frequency (float): Base frequency in Hz (20-20000)
            intensity (float): Volume/intensity (0.0 to 1.0)
            pattern (list): Optional pattern for complex audio [duration1, pause1, duration2, ...]
            preset (str): Optional preset name (overrides other parameters)
            
        Returns:
            Dictionary with:
                - audio_data: base64 encoded WAV audio
                - duration: actual duration in seconds
                - sample_rate: sample rate used
                - description: text description of audio
        """
        # Apply preset if specified
        if preset:
            audio_type, duration, frequency, intensity, pattern = self._apply_preset(preset)
        
        # Validate parameters
        intensity = max(0.0, min(1.0, intensity))
        frequency = max(20.0, min(20000.0, frequency))
        duration = max(0.01, min(10.0, duration))  # Limit to 10 seconds
        
        # Generate audio based on type
        if audio_type == self.AUDIO_TYPE_BEEP:
            audio_samples = self._generate_beep(duration, frequency, intensity)
            description = f"Beep at {frequency}Hz for {duration}s"
        
        elif audio_type == self.AUDIO_TYPE_TONE:
            audio_samples = self._generate_tone(duration, frequency, intensity)
            description = f"Tone at {frequency}Hz for {duration}s"
        
        elif audio_type == self.AUDIO_TYPE_SWEEP:
            end_freq = frequency * 2  # Sweep to double frequency
            audio_samples = self._generate_sweep(duration, frequency, end_freq, intensity)
            description = f"Sweep from {frequency}Hz to {end_freq}Hz"
        
        elif audio_type == self.AUDIO_TYPE_PULSE:
            audio_samples = self._generate_pulse(duration, frequency, intensity)
            description = f"Pulsing tone at {frequency}Hz"
        
        elif audio_type == self.AUDIO_TYPE_GEIGER:
            audio_samples = self._generate_geiger(duration, intensity)
            description = f"Geiger counter-style clicks (intensity: {intensity})"
        
        else:
            # Default to beep
            audio_samples = self._generate_beep(duration, frequency, intensity)
            description = f"Default beep at {frequency}Hz"
        
        # Apply pattern if specified
        if pattern:
            audio_samples = self._apply_pattern(audio_samples, pattern)
        
        # Convert to WAV format
        wav_data = self._samples_to_wav(audio_samples)
        
        # Encode to base64
        audio_b64 = base64.b64encode(wav_data).decode('utf-8')
        
        return {
            "audio_data": audio_b64,
            "duration": len(audio_samples) / self.SAMPLE_RATE,
            "sample_rate": self.SAMPLE_RATE,
            "description": description,
            "format": "wav"
        }
    
    def _apply_preset(self, preset: str) -> Tuple[str, float, float, float, Optional[List[float]]]:
        """
        Apply predefined presets for common use cases
        
        Returns:
            tuple: (audio_type, duration, frequency, intensity, pattern)
        """
        presets = {
            self.PRESET_SCANNING: (
                self.AUDIO_TYPE_GEIGER, 0.05, 800.0, 0.3, None
            ),
            self.PRESET_PROXIMITY: (
                self.AUDIO_TYPE_PULSE, 0.5, 440.0, 0.5, None
            ),
            self.PRESET_ALIGNMENT: (
                self.AUDIO_TYPE_SWEEP, 0.3, 300.0, 0.4, None
            ),
            self.PRESET_SUCCESS: (
                self.AUDIO_TYPE_TONE, 0.2, 880.0, 0.6, [0.1, 0.05, 0.1]
            ),
            self.PRESET_WARNING: (
                self.AUDIO_TYPE_BEEP, 0.15, 600.0, 0.7, [0.15, 0.1, 0.15]
            ),
            self.PRESET_ERROR: (
                self.AUDIO_TYPE_TONE, 0.4, 220.0, 0.8, None
            )
        }
        
        return presets.get(preset, (self.AUDIO_TYPE_BEEP, 0.2, 440.0, 0.5, None))
    
    def _generate_beep(self, duration: float, frequency: float, intensity: float) -> np.ndarray:
        """
        Generate a simple beep with envelope
        """
        num_samples = int(duration * self.SAMPLE_RATE)
        t = np.linspace(0, duration, num_samples, False)
        
        # Generate sine wave
        audio = np.sin(2 * np.pi * frequency * t)
        
        # Apply envelope (attack, sustain, release)
        envelope = self._create_envelope(num_samples, attack=0.01, release=0.05)
        audio = audio * envelope * intensity
        
        return audio
    
    def _generate_tone(self, duration: float, frequency: float, intensity: float) -> np.ndarray:
        """
        Generate a sustained tone
        """
        num_samples = int(duration * self.SAMPLE_RATE)
        t = np.linspace(0, duration, num_samples, False)
        
        # Generate sine wave
        audio = np.sin(2 * np.pi * frequency * t) * intensity
        
        return audio
    
    def _generate_sweep(
        self, 
        duration: float, 
        start_freq: float, 
        end_freq: float, 
        intensity: float
    ) -> np.ndarray:
        """
        Generate a frequency sweep (chirp)
        """
        num_samples = int(duration * self.SAMPLE_RATE)
        t = np.linspace(0, duration, num_samples, False)
        
        # Linear frequency sweep
        freq_sweep = np.linspace(start_freq, end_freq, num_samples)
        phase = 2 * np.pi * np.cumsum(freq_sweep) / self.SAMPLE_RATE
        audio = np.sin(phase) * intensity
        
        return audio
    
    def _generate_pulse(self, duration: float, frequency: float, intensity: float) -> np.ndarray:
        """
        Generate a pulsing tone (amplitude modulated)
        """
        num_samples = int(duration * self.SAMPLE_RATE)
        t = np.linspace(0, duration, num_samples, False)
        
        # Carrier frequency
        carrier = np.sin(2 * np.pi * frequency * t)
        
        # Modulation (8 Hz pulse)
        modulation = (np.sin(2 * np.pi * 8 * t) + 1) / 2
        
        audio = carrier * modulation * intensity
        
        return audio
    
    def _generate_geiger(self, duration: float, intensity: float) -> np.ndarray:
        """
        Generate geiger counter-style clicks
        
        Click rate varies with intensity (0.1 to 1.0)
        Low intensity = slow clicks, high intensity = rapid clicks
        """
        num_samples = int(duration * self.SAMPLE_RATE)
        audio = np.zeros(num_samples)
        
        # Click rate based on intensity (1 to 20 clicks per second)
        clicks_per_second = 1 + (intensity * 19)
        click_interval = int(self.SAMPLE_RATE / clicks_per_second)
        
        # Generate clicks
        click_samples = int(0.001 * self.SAMPLE_RATE)  # 1ms click duration
        
        position = 0
        while position < num_samples - click_samples:
            # Random click variation for realism
            actual_interval = int(click_interval * (0.8 + np.random.random() * 0.4))
            position += actual_interval
            
            if position + click_samples < num_samples:
                # Create click (short noise burst)
                click = np.random.randn(click_samples) * 0.8
                # Envelope for click
                click_env = np.exp(-np.linspace(0, 10, click_samples))
                audio[position:position + click_samples] = click * click_env
        
        return audio * intensity
    
    def _create_envelope(
        self, 
        num_samples: int, 
        attack: float = 0.01, 
        release: float = 0.05
    ) -> np.ndarray:
        """
        Create an ADSR-style envelope for audio
        
        Args:
            num_samples: Total number of samples
            attack: Attack time in seconds
            release: Release time in seconds
            
        Returns:
            Envelope array
        """
        envelope = np.ones(num_samples)
        
        attack_samples = int(attack * self.SAMPLE_RATE)
        release_samples = int(release * self.SAMPLE_RATE)
        
        # Attack (fade in)
        if attack_samples > 0 and attack_samples < num_samples:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Release (fade out)
        if release_samples > 0 and release_samples < num_samples:
            envelope[-release_samples:] = np.linspace(1, 0, release_samples)
        
        return envelope
    
    def _apply_pattern(self, audio: np.ndarray, pattern: List[float]) -> np.ndarray:
        """
        Apply a pattern to audio (repeat with pauses)
        
        Args:
            audio: Base audio samples
            pattern: List of [duration1, pause1, duration2, pause2, ...]
            
        Returns:
            Patterned audio
        """
        result = []
        
        for i, value in enumerate(pattern):
            if i % 2 == 0:
                # Audio segment
                segment_samples = int(value * self.SAMPLE_RATE)
                if segment_samples <= len(audio):
                    result.append(audio[:segment_samples])
                else:
                    result.append(audio)
            else:
                # Pause
                pause_samples = int(value * self.SAMPLE_RATE)
                result.append(np.zeros(pause_samples))
        
        return np.concatenate(result) if result else audio
    
    def _samples_to_wav(self, audio_samples: np.ndarray) -> bytes:
        """
        Convert audio samples to WAV format bytes
        
        Args:
            audio_samples: Numpy array of audio samples (float, -1 to 1)
            
        Returns:
            WAV format bytes
        """
        # Normalize to int16 range
        audio_int16 = (audio_samples * 32767).astype(np.int16)
        
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        
        # WAV header
        num_samples = len(audio_int16)
        num_channels = 1  # Mono
        sample_width = 2  # 16-bit = 2 bytes
        
        # Write WAV header
        wav_buffer.write(b'RIFF')
        wav_buffer.write(struct.pack('<I', 36 + num_samples * num_channels * sample_width))
        wav_buffer.write(b'WAVE')
        
        # Format chunk
        wav_buffer.write(b'fmt ')
        wav_buffer.write(struct.pack('<I', 16))  # Chunk size
        wav_buffer.write(struct.pack('<H', 1))   # Audio format (1 = PCM)
        wav_buffer.write(struct.pack('<H', num_channels))
        wav_buffer.write(struct.pack('<I', self.SAMPLE_RATE))
        wav_buffer.write(struct.pack('<I', self.SAMPLE_RATE * num_channels * sample_width))
        wav_buffer.write(struct.pack('<H', num_channels * sample_width))
        wav_buffer.write(struct.pack('<H', sample_width * 8))
        
        # Data chunk
        wav_buffer.write(b'data')
        wav_buffer.write(struct.pack('<I', num_samples * num_channels * sample_width))
        wav_buffer.write(audio_int16.tobytes())
        
        return wav_buffer.getvalue()
    
    def generate_proximity_feedback(
        self, 
        distance: float, 
        min_distance: float = 0.0, 
        max_distance: float = 1.0
    ) -> Dict:
        """
        Helper: Generate audio feedback based on proximity/distance
        
        Args:
            distance: Current distance (normalized 0-1 or actual value)
            min_distance: Minimum distance threshold
            max_distance: Maximum distance threshold
            
        Returns:
            Audio feedback dictionary
        """
        # Normalize distance to 0-1 range
        normalized_distance = (distance - min_distance) / (max_distance - min_distance)
        normalized_distance = max(0.0, min(1.0, normalized_distance))
        
        # Closer = higher frequency and faster clicks
        intensity = 1.0 - normalized_distance
        frequency = 300 + (intensity * 500)  # 300Hz to 800Hz
        
        return self.generate_audio_feedback(
            audio_type=self.AUDIO_TYPE_GEIGER,
            duration=0.1,
            frequency=frequency,
            intensity=intensity
        )
    
    def generate_alignment_feedback(
        self, 
        offset: float, 
        aligned_threshold: float = 0.1
    ) -> Dict:
        """
        Helper: Generate audio feedback for alignment status
        
        Args:
            offset: Offset from target (-1 to 1, where 0 is perfect)
            aligned_threshold: Threshold for considering aligned
            
        Returns:
            Audio feedback dictionary
        """
        abs_offset = abs(offset)
        
        if abs_offset <= aligned_threshold:
            # Aligned - success tone
            return self.generate_audio_feedback(preset=self.PRESET_SUCCESS)
        else:
            # Not aligned - sweep based on offset
            intensity = min(1.0, abs_offset)
            return self.generate_audio_feedback(
                audio_type=self.AUDIO_TYPE_SWEEP,
                duration=0.2,
                frequency=300 + (abs_offset * 300),
                intensity=0.5
            )
    
    def generate_status_feedback(self, status: str) -> Dict:
        """
        Helper: Generate audio feedback for status events
        
        Args:
            status: Status string (success, warning, error, scanning, etc.)
            
        Returns:
            Audio feedback dictionary
        """
        status_lower = status.lower()
        
        if "success" in status_lower or "complete" in status_lower:
            return self.generate_audio_feedback(preset=self.PRESET_SUCCESS)
        elif "warning" in status_lower or "caution" in status_lower:
            return self.generate_audio_feedback(preset=self.PRESET_WARNING)
        elif "error" in status_lower or "fail" in status_lower:
            return self.generate_audio_feedback(preset=self.PRESET_ERROR)
        elif "scan" in status_lower or "search" in status_lower:
            return self.generate_audio_feedback(preset=self.PRESET_SCANNING)
        else:
            return self.generate_audio_feedback(preset=self.PRESET_PROXIMITY)


processor = AudioFeedbackProcessor()
app = processor.app
