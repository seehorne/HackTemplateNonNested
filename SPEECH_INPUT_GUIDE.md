# Speech Input Building Block - Developer Guide

This guide explains how to use the Speech Input Building Block in your custom processors.

## Overview

The Speech Input Building Block provides CPU-only speech-to-text capabilities using OpenAI's Whisper model (via the distil-whisper variant for faster inference). It's designed to be easily integrated into any processor that needs voice input functionality.

## Features

- **CPU-Only**: No GPU required, uses distil-whisper for efficient CPU inference
- **Easy Integration**: Simple API with `SpeechInputHelper` class
- **Voice Commands**: Optional `VoiceCommandProcessor` for command detection
- **Standalone or Embedded**: Can be used as a standalone service or imported as a module
- **Building Block Design**: Designed to be reused across multiple processors

## Architecture

The speech input system consists of three main components:

1. **speech_input_processor.py** - Standalone FastAPI processor
2. **speech_input_building_block.py** - Reusable helper module
3. **voice_controlled_example_processor.py** - Example integration

## Quick Start

### Option 1: Using as a Standalone Service

Start the speech input processor:

```bash
conda activate whatsai
uvicorn processors.speech_input_processor:app --host 127.0.0.1 --port 8014
```

Send audio for transcription:

```python
import requests
import base64

# Load your audio file
with open("audio.wav", "rb") as f:
    audio_bytes = f.read()

# Send to processor
response = requests.post(
    "http://127.0.0.1:8014/transcribe",
    json={
        "audio": base64.b64encode(audio_bytes).decode(),
        "format": "raw"
    }
)

print(response.json()["transcription"])
```

### Option 2: Import as a Building Block

Create a custom processor that uses speech input:

```python
from processors.speech_input_building_block import SpeechInputHelper
from processors.base_processor import BaseProcessor

class MyVoiceProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        # Add speech capability
        self.speech = SpeechInputHelper()
    
    def process_frame(self, frame):
        # Your image processing logic
        return frame, "Processed"
    
    def process_with_audio(self, frame, audio_data):
        # Transcribe audio
        text = self.speech.transcribe(audio_data)
        
        # Use transcription in processing
        result = f"You said: {text}"
        return frame, result
```

## Integration Examples

### Example 1: Simple Transcription

```python
from processors.speech_input_building_block import SpeechInputHelper

# Initialize
speech = SpeechInputHelper()

# Check if available
if speech.is_available():
    # Transcribe audio
    text = speech.transcribe(audio_bytes)
    print(f"Transcription: {text}")
```

### Example 2: Voice Commands

```python
from processors.speech_input_building_block import (
    SpeechInputHelper,
    VoiceCommandProcessor
)

# Initialize
speech = SpeechInputHelper()
commands = VoiceCommandProcessor()

# Define commands
def zoom_in():
    print("Zooming in...")

def zoom_out():
    print("Zooming out...")

commands.add_command("zoom in", zoom_in)
commands.add_command("zoom out", zoom_out)

# Process audio
text = speech.transcribe(audio_bytes)
commands.execute_command(text)
```

### Example 3: With Timestamps

```python
from processors.speech_input_building_block import SpeechInputHelper

speech = SpeechInputHelper()

# Get transcription with word-level timestamps
result = speech.transcribe_with_timestamps(audio_bytes)

print(f"Text: {result['text']}")
for chunk in result['chunks']:
    print(f"  [{chunk['timestamp']}] {chunk['text']}")
```

## API Reference

### SpeechInputHelper

Main class for speech-to-text functionality.

#### Constructor

```python
SpeechInputHelper(
    model_name: str = "distil-whisper/distil-small.en",
    device: str = "cpu"
)
```

**Parameters:**
- `model_name`: HuggingFace model identifier (default: distil-whisper/distil-small.en)
- `device`: Computation device (default: "cpu")

#### Methods

**`is_available() -> bool`**

Check if speech recognition is available.

**`transcribe(audio_data: Union[bytes, np.ndarray]) -> str`**

Transcribe audio to text.

- **Parameters:** `audio_data` - Audio as bytes or numpy array
- **Returns:** Transcribed text string

**`transcribe_with_timestamps(audio_data: Union[bytes, np.ndarray]) -> dict`**

Transcribe with word-level timestamps.

- **Parameters:** `audio_data` - Audio as bytes or numpy array
- **Returns:** Dict with "text" and "chunks" keys

### VoiceCommandProcessor

Helper for processing voice commands.

#### Constructor

```python
VoiceCommandProcessor(commands: Optional[dict] = None)
```

**Parameters:**
- `commands`: Dictionary mapping keywords to action functions

#### Methods

**`add_command(keyword: str, action: callable)`**

Register a voice command.

**`process_text(text: str) -> Optional[str]`**

Detect command in text. Returns keyword or None.

**`execute_command(text: str, *args, **kwargs)`**

Detect and execute command. Returns action result or None.

## Audio Format

The processor expects audio in the following format:

- **Sample Rate**: 16 kHz (Whisper's native rate)
- **Format**: float32 PCM or int16 PCM
- **Channels**: Mono (1 channel)
- **Encoding**: Base64 for HTTP endpoints, bytes or numpy array for Python API

### Converting Audio

```python
import numpy as np
import librosa

# Load audio file and resample to 16kHz
audio, sr = librosa.load("audio.wav", sr=16000, mono=True)

# Convert to float32 if needed
audio = audio.astype(np.float32)

# Now ready for transcription
text = speech.transcribe(audio)
```

## Configuration

The speech input processor is configured in `processor_config.json`:

```json
{
  "13": {
    "host": "127.0.0.1",
    "port": 8014,
    "name": "speech_input_processor",
    "conda_env": "whatsai",
    "dependencies": [],
    "expects_input": "audio",
    "description": "Building block processor for speech-to-text",
    "enabled": true
  }
}
```

## Performance Considerations

- **First Run**: Model download may take 1-2 minutes on first use
- **Inference Speed**: ~1-2x realtime on modern CPUs (distil-whisper is optimized)
- **Memory Usage**: ~1-2 GB for distil-whisper-small.en model
- **Accuracy**: Good for English speech, may struggle with heavy accents

## Troubleshooting

### Model Not Loading

If the model fails to load:

```python
# Check if transformers is installed
pip install transformers torch

# Manually download model
from transformers import pipeline
model = pipeline("automatic-speech-recognition", 
                model="distil-whisper/distil-small.en")
```

### Audio Format Issues

If transcription fails:

```python
# Ensure proper audio format
import numpy as np

# Convert int16 to float32
if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0

# Ensure mono
if len(audio.shape) > 1:
    audio = audio.mean(axis=1)

# Ensure 16kHz
# Use librosa to resample if needed
```

### Low Accuracy

If transcription accuracy is poor:

- Ensure audio is clear and at proper volume
- Check sample rate is 16kHz
- Consider using a larger Whisper model (trade-off: slower inference)
- Reduce background noise in audio

## Future Processors

When creating new processors that need speech input:

1. Import `SpeechInputHelper` from `processors.speech_input_building_block`
2. Initialize in your processor's `__init__` method
3. Use `transcribe()` method to convert audio to text
4. Optionally use `VoiceCommandProcessor` for command handling

See `processors/voice_controlled_example_processor.py` for a complete example.

## License

This building block is part of the WhatsAI project and follows the same license.

## References

- OpenAI Whisper: https://github.com/openai/whisper
- Distil-Whisper: https://huggingface.co/distil-whisper
- Transformers: https://huggingface.co/docs/transformers
