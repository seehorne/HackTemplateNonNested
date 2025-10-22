# Speech Input Processor - Quick Reference

## Installation

```bash
# Install Vosk
conda run -n whatsai pip install vosk

# Download model (English small)
cd models && mkdir -p vosk && cd vosk
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
```

## Basic Usage

```python
from processors.speech_input_processor import SpeechInputProcessor

# Initialize
speech = SpeechInputProcessor(language="en-us", model_size="small")

# Recognize speech
result = speech.recognize_speech(audio_bytes)

# Check result
if result['success']:
    print(f"Text: {result['text']}")
    print(f"Confidence: {result['confidence']}")
```

## Voice Commands

```python
# Create command handler
commands = {
    "start": lambda: start_action(),
    "stop": lambda: stop_action(),
    "reset": lambda: reset_action()
}

handler = speech.create_voice_command_handler(commands)

# Process audio
result = handler(audio_data)
if result['executed']:
    print(f"Executed: {result['command']}")
```

## Streaming Recognition

```python
# Process audio chunks
results = speech.recognize_speech_streaming(
    audio_chunks,
    get_partial=True
)

for result in results:
    if result['success']:
        print(result['text'])
```

## Building Block Integration

```python
class MyProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.speech = SpeechInputProcessor()
    
    def process_audio(self, audio_data):
        result = self.speech.recognize_speech(audio_data)
        return result['text'] if result['success'] else ""
```

## Result Format

```python
{
    "text": "recognized text",
    "confidence": 0.95,
    "words": [
        {"word": "recognized", "start": 0.0, "end": 0.5, "conf": 0.96},
        {"word": "text", "start": 0.5, "end": 0.8, "conf": 0.94}
    ],
    "partial": False,
    "success": True
}
```

## Audio Format

- **Sample Rate:** 16000 Hz
- **Channels:** Mono (1)
- **Bit Depth:** 16-bit
- **Format:** WAV or raw PCM

## Available Languages

| Code | Language |
|------|----------|
| en-us | English (US) |
| en-gb | English (UK) |
| es | Spanish |
| fr | French |
| de | German |
| it | Italian |
| pt | Portuguese |
| ru | Russian |
| zh | Chinese |
| ja | Japanese |

## Common Patterns

### 1. Command Recognition
```python
result = speech.recognize_speech(audio_data)
text = result['text'].lower()

if 'start' in text:
    start_process()
elif 'stop' in text:
    stop_process()
```

### 2. With Audio Feedback
```python
from processors.audio_feedback_processor import AudioFeedbackProcessor

audio_fb = AudioFeedbackProcessor()
result = speech.recognize_speech(audio_data)

if result['success']:
    feedback = audio_fb.generate_audio_feedback(preset="success")
else:
    feedback = audio_fb.generate_audio_feedback(preset="error")
```

### 3. Dictation Mode
```python
text_buffer = []

for chunk in audio_chunks:
    result = speech.recognize_speech(chunk, mode="dictation")
    if result['success']:
        text_buffer.append(result['text'])

full_text = " ".join(text_buffer)
```

## Tips

✅ **Do:**
- Use small models for commands
- Check `result['success']` before using text
- Normalize text (lowercase, strip) for matching
- Handle errors gracefully
- Provide user feedback

❌ **Don't:**
- Assume recognition is perfect
- Process very long audio at once
- Ignore confidence scores
- Mix languages in same processor

## Troubleshooting

**Model not found?**
```bash
# Download and extract model to models/vosk/
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip -d models/vosk/
```

**Poor accuracy?**
- Use larger model
- Improve audio quality
- Reduce background noise
- Speak clearly

**Slow performance?**
- Use smaller model
- Process shorter chunks
- Enable partial results

## See Also

- Full documentation: `docs/SPEECH_INPUT_PROCESSOR.md`
- Example processor: `processors/speech_command_example_processor.py`
- Vosk models: https://alphacephei.com/vosk/models
