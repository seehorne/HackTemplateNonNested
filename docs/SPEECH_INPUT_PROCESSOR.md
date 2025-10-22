# Speech Input Building Block Processor

## Overview

The Speech Input Processor is a reusable building block that provides speech-to-text (voice recognition) capabilities for accessibility and voice-based interactions. It's designed to be integrated into other processors that need voice input capabilities.

## Features

- **CPU-Only**: Uses Vosk for offline speech recognition, no GPU required
- **Offline Recognition**: Works without internet connection once models are downloaded
- **Multiple Languages**: Support for 10+ languages (English, Spanish, French, German, etc.)
- **Word-Level Timestamps**: Detailed timing information for each recognized word
- **Confidence Scores**: Per-word and overall confidence metrics
- **Multiple Modes**: Single command, continuous, and dictation modes
- **Building Block Design**: Easy integration into other processors
- **Voice Command Handler**: Built-in utility for creating voice-controlled applications

## Supported Languages

- English (US and UK): `en-us`, `en-gb`
- Spanish: `es`
- French: `fr`
- German: `de`
- Italian: `it`
- Portuguese: `pt`
- Russian: `ru`
- Chinese: `zh`
- Japanese: `ja`

*Additional languages can be added by downloading appropriate Vosk models.*

## Recognition Modes

### 1. Single Command
Best for discrete voice commands (e.g., "start", "stop", "reset").
- **Use Case**: Button-like interactions, menu navigation
- **Behavior**: Processes complete utterances

### 2. Continuous
Best for ongoing speech recognition with multiple commands.
- **Use Case**: Conversational interfaces, multi-step instructions
- **Behavior**: Continuously recognizes speech segments

### 3. Dictation
Best for transcribing longer speech segments.
- **Use Case**: Note-taking, form filling, document creation
- **Behavior**: Optimized for longer utterances with punctuation

## Installation

### 1. Install Vosk Library

The Vosk library should already be included in `pyproject.toml`:

```toml
dependencies = [
    ...
    "vosk"
]
```

Install dependencies:
```bash
conda run -n whatsai pip install -e resources/whatsai/
```

### 2. Download Speech Model

Download a Vosk model for your language:

**For English (lightweight, ~50MB):**
```bash
cd models
mkdir -p vosk
cd vosk
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
```

**For better accuracy (larger, ~1.5GB):**
```bash
wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip
unzip vosk-model-en-us-0.22.zip
```

**For other languages:**
Visit [https://alphacephei.com/vosk/models](https://alphacephei.com/vosk/models) and download the appropriate model.

## Usage

### As a Standalone Processor

```python
from processors.speech_input_processor import SpeechInputProcessor

# Initialize
speech_proc = SpeechInputProcessor(
    language="en-us",
    model_size="small"  # or "large" for better accuracy
)

# Recognize speech from audio data
result = speech_proc.recognize_speech(audio_bytes)

print(f"Text: {result['text']}")
print(f"Confidence: {result['confidence']}")
print(f"Words: {result['words']}")
```

### As a Building Block in Other Processors

```python
from processors.speech_input_processor import SpeechInputProcessor
from processors.base_processor import BaseProcessor

class MyVoiceControlledProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        # Initialize speech input as a building block
        self.speech_input = SpeechInputProcessor(
            language="en-us",
            model_size="small"
        )
    
    def process_audio_command(self, audio_data):
        # Recognize speech
        result = self.speech_input.recognize_speech(audio_data)
        
        if result['success']:
            text = result['text'].lower()
            
            # Handle commands based on recognized text
            if 'start' in text:
                self.start_processing()
            elif 'stop' in text:
                self.stop_processing()
            
            return f"Command recognized: {text}"
        else:
            return "Command not recognized"
```

### Using Voice Command Handler

The processor includes a built-in voice command handler utility:

```python
from processors.speech_input_processor import SpeechInputProcessor

speech_input = SpeechInputProcessor()

# Define commands and their callbacks
commands = {
    "start": lambda: print("Starting..."),
    "stop": lambda: print("Stopping..."),
    "reset": lambda: print("Resetting..."),
    "help": lambda: print("Available commands: start, stop, reset, help")
}

# Create command handler
handler = speech_input.create_voice_command_handler(commands)

# Use handler with audio data
result = handler(audio_bytes)

if result['executed']:
    print(f"Executed: {result['command']}")
else:
    print(f"No matching command for: {result['text']}")
```

## API Reference

### Main Methods

#### `recognize_speech(audio_data, mode, partial_results)`

Recognize speech from audio data.

**Parameters:**
- `audio_data` (bytes or np.ndarray): Audio data as WAV bytes or PCM 16-bit array
- `mode` (str): Recognition mode - "single_command", "continuous", or "dictation"
- `partial_results` (bool): Return partial results during recognition

**Returns:**
```python
{
    "text": "recognized text",
    "confidence": 0.95,  # 0.0 to 1.0
    "words": [
        {"word": "recognized", "start": 0.0, "end": 0.5, "conf": 0.96},
        {"word": "text", "start": 0.5, "end": 0.8, "conf": 0.94}
    ],
    "partial": False,
    "success": True
}
```

#### `recognize_speech_streaming(audio_chunks, get_partial)`

Recognize speech from streaming audio chunks.

**Parameters:**
- `audio_chunks` (List[bytes]): List of audio data chunks
- `get_partial` (bool): Whether to include partial results

**Returns:** List of recognition results

#### `get_speech_from_base64_wav(wav_base64)`

Helper method to recognize speech from base64-encoded WAV data.

**Parameters:**
- `wav_base64` (str): Base64-encoded WAV audio

**Returns:** Recognition result dictionary

#### `create_voice_command_handler(commands)`

Create a voice command handler function.

**Parameters:**
- `commands` (Dict[str, callable]): Dictionary mapping command phrases to callback functions

**Returns:** Handler function that takes audio data and executes matching commands

## Audio Format Requirements

Vosk works best with the following audio format:

- **Sample Rate:** 16000 Hz (16 kHz)
- **Channels:** 1 (Mono)
- **Sample Width:** 16-bit
- **Format:** PCM or WAV

The processor will automatically extract PCM data from WAV files and handle basic format conversions.

## Building Block Integration Examples

### Example 1: Voice-Controlled Image Filter

See `processors/speech_command_example_processor.py` for a complete example:

```python
class SpeechCommandExampleProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.speech_input = SpeechInputProcessor()
        
        # Create voice command handler
        self.command_handler = self.speech_input.create_voice_command_handler({
            "grayscale": lambda: self._set_filter("grayscale"),
            "blur": lambda: self._set_filter("blur"),
            "edge": lambda: self._set_filter("edge"),
        })
    
    def process_audio(self, audio_data):
        result = self.command_handler(audio_data)
        return result
```

### Example 2: Voice Navigation

```python
class VoiceNavigationProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.speech_input = SpeechInputProcessor()
        self.position = {"x": 0, "y": 0}
    
    def process_voice_command(self, audio_data):
        result = self.speech_input.recognize_speech(audio_data)
        
        if result['success']:
            text = result['text'].lower()
            
            # Parse directional commands
            if 'left' in text:
                self.position['x'] -= 10
            elif 'right' in text:
                self.position['x'] += 10
            elif 'up' in text:
                self.position['y'] -= 10
            elif 'down' in text:
                self.position['y'] += 10
            
            return f"Position: {self.position}"
```

### Example 3: Voice Dictation

```python
class VoiceDictationProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.speech_input = SpeechInputProcessor()
        self.text_buffer = []
    
    def process_streaming_audio(self, audio_chunks):
        results = self.speech_input.recognize_speech_streaming(
            audio_chunks,
            get_partial=True
        )
        
        for result in results:
            if result['success'] and not result['partial']:
                self.text_buffer.append(result['text'])
        
        return " ".join(self.text_buffer)
```

## Performance Considerations

### Model Size Trade-offs

**Small Models (~50MB):**
- ✅ Faster loading and recognition
- ✅ Lower memory usage
- ✅ Good for simple commands
- ⚠️ Lower accuracy for complex speech

**Large Models (~1-2GB):**
- ✅ Higher accuracy
- ✅ Better with accents and background noise
- ✅ More robust word recognition
- ⚠️ Slower loading and processing
- ⚠️ Higher memory usage

### CPU Usage

- Speech recognition is CPU-intensive
- Processing time depends on audio length and model size
- For real-time applications, use small models
- Consider chunking long audio for better responsiveness

## Troubleshooting

### Model Not Found

**Error:** "Vosk model not found"

**Solution:**
1. Download the appropriate model from https://alphacephei.com/vosk/models
2. Extract to `models/vosk/vosk-model-small-en-us/` (or appropriate directory)
3. Ensure the model directory contains the required files

### Poor Recognition Quality

**Solutions:**
1. Use a larger model for better accuracy
2. Ensure audio quality is good (16kHz, 16-bit mono)
3. Reduce background noise
4. Speak clearly and at normal pace
5. Check microphone quality

### Slow Performance

**Solutions:**
1. Use smaller model
2. Reduce audio chunk size
3. Process in separate thread
4. Enable partial results for faster feedback

### Import Error

**Error:** "Vosk library not installed"

**Solution:**
```bash
conda run -n whatsai pip install vosk
```

## Best Practices

1. **Choose Appropriate Model:** Use small models for commands, large for dictation
2. **Handle Errors Gracefully:** Always check `result['success']` before using text
3. **Provide Feedback:** Give users audio/visual confirmation of recognized commands
4. **Use Confidence Scores:** Filter low-confidence results to reduce errors
5. **Normalize Text:** Convert to lowercase and strip whitespace for command matching
6. **Cache Model:** Load model once and reuse across multiple recognitions
7. **Chunk Long Audio:** Process long recordings in chunks for better responsiveness

## Integration with Other Building Blocks

### With Audio Feedback Processor

Combine speech input with audio feedback for voice-controlled interactions:

```python
from processors.speech_input_processor import SpeechInputProcessor
from processors.audio_feedback_processor import AudioFeedbackProcessor

class VoiceWithFeedbackProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.speech_input = SpeechInputProcessor()
        self.audio_feedback = AudioFeedbackProcessor()
    
    def process_voice_command(self, audio_data):
        # Recognize command
        result = self.speech_input.recognize_speech(audio_data)
        
        # Provide audio feedback
        if result['success']:
            feedback = self.audio_feedback.generate_audio_feedback(
                preset="success"
            )
            return {
                "text": result['text'],
                "audio": feedback
            }
        else:
            feedback = self.audio_feedback.generate_audio_feedback(
                preset="error"
            )
            return {
                "error": "Command not recognized",
                "audio": feedback
            }
```

### With Hand Tracking

Create gesture + voice controlled interfaces:

```python
from processors.speech_input_processor import SpeechInputProcessor
from processors.hand_tracking_processor import HandTrackingProcessor

class MultimodalInputProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.speech_input = SpeechInputProcessor()
        self.hand_tracker = HandTrackingProcessor()
    
    def process_multimodal(self, frame, audio_data):
        # Get hand position
        hand_data = self.hand_tracker.get_hand_tracking_data(frame)
        
        # Get voice command
        speech_result = self.speech_input.recognize_speech(audio_data)
        
        # Combine inputs
        return {
            "hand_position": hand_data['hands'][0]['location'] if hand_data['hands'] else None,
            "voice_command": speech_result['text'] if speech_result['success'] else None
        }
```

## Future Enhancements

Potential future additions:

- Wake word detection (e.g., "Hey Assistant")
- Speaker identification
- Real-time streaming mode
- Automatic language detection
- Custom vocabulary support
- Integration with cloud speech services as fallback

## License

This processor uses Vosk, which is licensed under Apache 2.0 License.

## References

- Vosk Speech Recognition: https://alphacephei.com/vosk/
- Vosk GitHub: https://github.com/alphacep/vosk-api
- Vosk Models: https://alphacephei.com/vosk/models
