# Speech Input Building Block - Quick Reference

## For Developers Creating New Processors

### Minimal Integration (Just Transcription)

```python
from processors.speech_input_building_block import SpeechInputHelper
from processors.base_processor import BaseProcessor

class MyProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.speech = SpeechInputHelper()
    
    def process_frame(self, frame):
        # Your image processing
        return frame, "result"
    
    # Add custom endpoint for audio
    def process_audio(self, audio_bytes):
        text = self.speech.transcribe(audio_bytes)
        return text
```

### With Voice Commands

```python
from processors.speech_input_building_block import (
    SpeechInputHelper, 
    VoiceCommandProcessor
)

class MyProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.speech = SpeechInputHelper()
        
        # Setup commands
        self.commands = VoiceCommandProcessor()
        self.commands.add_command("start", self.on_start)
        self.commands.add_command("stop", self.on_stop)
        self.commands.add_command("reset", self.on_reset)
    
    def on_start(self):
        print("Starting...")
    
    def on_stop(self):
        print("Stopping...")
    
    def on_reset(self):
        print("Resetting...")
    
    def process_with_voice(self, frame, audio_bytes):
        # Get transcription
        text = self.speech.transcribe(audio_bytes)
        
        # Execute command if detected
        self.commands.execute_command(text)
        
        return frame, f"Heard: {text}"
```

## API Quick Reference

### SpeechInputHelper

```python
# Initialize (CPU-only by default)
speech = SpeechInputHelper()

# Check if model loaded
if speech.is_available():
    # Transcribe audio bytes
    text = speech.transcribe(audio_bytes)
    
    # Or with timestamps
    result = speech.transcribe_with_timestamps(audio_bytes)
    print(result['text'])
    for chunk in result['chunks']:
        print(f"{chunk['timestamp']}: {chunk['text']}")
```

### VoiceCommandProcessor

```python
# Initialize
commands = VoiceCommandProcessor()

# Add commands
commands.add_command("zoom in", zoom_in_function)
commands.add_command("zoom out", zoom_out_function)

# Process text
command = commands.process_text("please zoom in now")  # Returns "zoom in"

# Or execute directly
commands.execute_command("zoom in please")  # Calls zoom_in_function()
```

## Processor Configuration

Add to `processor_config.json`:

```json
{
  "ID": {
    "host": "127.0.0.1",
    "port": 80XX,
    "name": "your_processor_name",
    "conda_env": "whatsai",
    "dependencies": [13],  // Depends on speech_input_processor
    "expects_input": "image+audio",
    "description": "Your description",
    "enabled": true
  }
}
```

## Audio Format Requirements

- **Sample Rate**: 16 kHz
- **Format**: float32 or int16 PCM
- **Channels**: Mono
- **Input**: bytes or numpy array

```python
# Example: Convert audio to correct format
import numpy as np

# If int16, convert to float32
if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0

# If stereo, convert to mono
if len(audio.shape) > 1:
    audio = audio.mean(axis=1)
```

## Dependencies

Add to your environment's `pyproject.toml`:

```toml
dependencies = [
    "transformers>=4.51.3",
    "torch",
    "torchaudio",
    "accelerate",
    "librosa"
]
```

## Testing

```bash
# Start your processor
uvicorn processors.your_processor:app --host 127.0.0.1 --port 80XX

# Test with the provided test script
python3 test_speech_input_processor.py
```

## Common Patterns

### Pattern 1: Audio + Image Processing

```python
def process_frame_with_voice(self, frame, audio_data):
    # Get voice input
    command = self.speech.transcribe(audio_data)
    
    # Modify processing based on voice
    if "enhance" in command.lower():
        frame = self.enhance(frame)
    elif "blur" in command.lower():
        frame = self.blur(frame)
    
    return frame, f"Applied: {command}"
```

### Pattern 2: Continuous Voice Control

```python
class StreamingVoiceProcessor:
    def __init__(self):
        self.speech = SpeechInputHelper()
        self.state = "idle"
    
    async def handle_audio_stream(self, audio_chunks):
        for chunk in audio_chunks:
            text = self.speech.transcribe(chunk)
            
            # Update state based on voice
            if "start" in text.lower():
                self.state = "active"
            elif "stop" in text.lower():
                self.state = "idle"
```

### Pattern 3: Voice-Activated Features

```python
def process_frame(self, frame, audio_data=None):
    # Normal processing
    result = self.detect_objects(frame)
    
    # Add voice-activated features
    if audio_data:
        command = self.speech.transcribe(audio_data)
        
        # Voice-activated zoom
        if "zoom" in command.lower():
            frame = self.zoom(frame)
        
        # Voice-activated filter
        if "filter" in command.lower():
            frame = self.apply_filter(frame)
    
    return frame, result
```

## Full Example

See `processors/voice_controlled_example_processor.py` for a complete working example.

## Troubleshooting

**Model not loading**: Ensure transformers and torch are installed
**Low accuracy**: Check audio quality and sample rate (should be 16kHz)
**Slow inference**: Use distil-whisper models (already default)
**Memory issues**: Reduce batch size or use smaller model variant

## More Information

See `SPEECH_INPUT_GUIDE.md` for comprehensive documentation.
