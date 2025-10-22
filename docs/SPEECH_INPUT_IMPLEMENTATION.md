# Speech Input Processor - Implementation Guide

This guide explains the internal implementation details of the Speech Input Processor for developers who want to understand or extend the processor.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                  Speech Input Processor                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │        Audio Input Layer                            │    │
│  │  - WAV format support                              │    │
│  │  - PCM byte array support                          │    │
│  │  - Base64 encoded audio support                    │    │
│  └────────────────────────────────────────────────────┘    │
│                        ↓                                     │
│  ┌────────────────────────────────────────────────────┐    │
│  │        Audio Preprocessing                          │    │
│  │  - Format conversion                               │    │
│  │  - Sample rate verification (16kHz)                │    │
│  │  - Channel conversion (mono)                       │    │
│  │  - Bit depth normalization (16-bit)                │    │
│  └────────────────────────────────────────────────────┘    │
│                        ↓                                     │
│  ┌────────────────────────────────────────────────────┐    │
│  │        Vosk Recognition Engine                      │    │
│  │  - Model loading (lazy initialization)             │    │
│  │  - Speech-to-text conversion                       │    │
│  │  - Word-level timestamps                           │    │
│  │  - Confidence scoring                              │    │
│  └────────────────────────────────────────────────────┘    │
│                        ↓                                     │
│  ┌────────────────────────────────────────────────────┐    │
│  │        Result Processing                            │    │
│  │  - Text extraction                                 │    │
│  │  - Confidence calculation                          │    │
│  │  - Word timing information                         │    │
│  │  - Result formatting                               │    │
│  └────────────────────────────────────────────────────┘    │
│                        ↓                                     │
│  ┌────────────────────────────────────────────────────┐    │
│  │        Optional: Command Handler                    │    │
│  │  - Command matching                                │    │
│  │  - Callback execution                              │    │
│  │  - Error handling                                  │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Model Management

The processor uses lazy loading for Vosk models to avoid unnecessary initialization:

```python
def _ensure_model_loaded(self):
    """Lazy load Vosk model on first use"""
    if self._model_loaded:
        return True
    
    if self._load_error:
        return False
    
    try:
        import vosk
        model_path = self._get_or_download_model()
        self.model = vosk.Model(model_path)
        self.recognizer = vosk.KaldiRecognizer(self.model, self.SAMPLE_RATE)
        self.recognizer.SetWords(True)  # Enable word-level timestamps
        self._model_loaded = True
        return True
    except Exception as e:
        self._load_error = str(e)
        return False
```

**Benefits:**
- Faster processor initialization
- Model only loaded when needed
- Graceful handling of missing models
- Clear error messages for users

### 2. Audio Format Handling

The processor accepts multiple audio formats and converts them to Vosk's required format:

```python
def _prepare_audio_data(self, audio_data):
    """Convert various audio formats to PCM 16-bit mono"""
    
    if isinstance(audio_data, np.ndarray):
        # NumPy array → bytes
        if audio_data.dtype != np.int16:
            audio_data = (audio_data * 32767).astype(np.int16)
        return audio_data.tobytes()
    
    elif isinstance(audio_data, bytes):
        if audio_data[:4] == b'RIFF':
            # WAV format → extract PCM
            return self._extract_pcm_from_wav(audio_data)
        else:
            # Assume raw PCM
            return audio_data
```

**Supported Formats:**
1. **WAV files** (bytes): Automatically extracts PCM data
2. **PCM byte arrays**: Used directly
3. **NumPy arrays**: Converted to int16 PCM

### 3. Recognition Modes

Three recognition modes are supported:

```python
# Mode constants
MODE_SINGLE_COMMAND = "single_command"
MODE_CONTINUOUS = "continuous"
MODE_DICTATION = "dictation"

def recognize_speech(self, audio_data, mode=MODE_SINGLE_COMMAND, partial_results=False):
    """
    Mode affects how audio is processed:
    - single_command: Wait for complete utterance
    - continuous: Process ongoing speech
    - dictation: Optimize for longer text
    """
    # Implementation handles mode-specific processing
```

### 4. Result Formatting

Results are standardized for consistent integration:

```python
def _format_result(self, result, partial=False):
    """
    Standardize Vosk results to consistent format
    """
    text = result.get("text", result.get("partial", ""))
    
    # Calculate average confidence from word results
    confidence = 0.0
    words = []
    
    if "result" in result:
        word_results = result["result"]
        if word_results:
            confidences = [w.get("conf", 0.0) for w in word_results]
            confidence = sum(confidences) / len(confidences)
            
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
        # No word-level data - estimate confidence
        confidence = 0.8 if text else 0.0
    
    return {
        "text": text,
        "confidence": confidence,
        "words": words,
        "partial": partial,
        "success": bool(text)
    }
```

### 5. Voice Command Handler

Utility for creating command-driven applications:

```python
def create_voice_command_handler(self, commands):
    """
    Create a handler that maps phrases to callbacks
    
    Args:
        commands: {"phrase": callback_function, ...}
    
    Returns:
        Handler function
    """
    def handler(audio_data):
        result = self.recognize_speech(audio_data)
        
        if not result["success"]:
            return {"command": None, "executed": False}
        
        text = result["text"].lower().strip()
        
        # Find matching command
        for cmd_phrase, callback in commands.items():
            if cmd_phrase.lower() in text:
                try:
                    callback()
                    return {
                        "command": cmd_phrase,
                        "text": text,
                        "executed": True
                    }
                except Exception as e:
                    return {
                        "command": cmd_phrase,
                        "executed": False,
                        "error": str(e)
                    }
        
        return {"command": None, "text": text, "executed": False}
    
    return handler
```

## Performance Considerations

### Model Size vs. Accuracy

| Model Size | Download Size | Memory Usage | Speed | Accuracy | Use Case |
|------------|--------------|--------------|-------|----------|----------|
| Small      | ~50MB        | ~200MB       | Fast  | Good     | Commands |
| Large      | ~1.5GB       | ~2GB         | Slower| Excellent| Dictation|

### CPU Usage

Speech recognition is CPU-intensive. Optimization strategies:

1. **Lazy Loading**: Model only loaded when needed
2. **Efficient Format Conversion**: Direct PCM handling when possible
3. **Streaming Support**: Process audio in chunks
4. **Partial Results**: Quick feedback without waiting for complete utterance

### Memory Management

```python
# Reset recognizer between utterances to free memory
self.recognizer.Reset()

# For long sessions, periodically reinitialize
if total_processed > REINIT_THRESHOLD:
    self._reinitialize_recognizer()
```

## Extension Points

### Adding New Languages

1. Download Vosk model for the language
2. Extract to `models/vosk/`
3. Add language code to `SUPPORTED_LANGUAGES`

```python
SUPPORTED_LANGUAGES = {
    "en-us": "English (US)",
    "es": "Spanish",
    # Add new language here
    "ko": "Korean"
}
```

### Custom Preprocessing

Extend `_prepare_audio_data` for custom audio formats:

```python
def _prepare_audio_data(self, audio_data):
    """Add custom format support"""
    
    # Existing format handling...
    
    # Add custom format
    if hasattr(audio_data, 'custom_format'):
        return self._convert_custom_format(audio_data)
    
    # ... rest of implementation
```

### Enhanced Command Matching

Extend command handler with fuzzy matching or NLP:

```python
from difflib import SequenceMatcher

def fuzzy_match_command(text, commands, threshold=0.8):
    """Fuzzy match commands for better recognition"""
    best_match = None
    best_score = 0.0
    
    for cmd in commands:
        score = SequenceMatcher(None, text, cmd).ratio()
        if score > best_score:
            best_score = score
            best_match = cmd
    
    return best_match if best_score >= threshold else None
```

## Testing

### Unit Tests

```python
def test_audio_format_conversion():
    """Test various audio format conversions"""
    proc = SpeechInputProcessor()
    
    # Test numpy array
    audio_np = np.zeros(16000, dtype=np.int16)
    pcm = proc._prepare_audio_data(audio_np)
    assert isinstance(pcm, bytes)
    
    # Test WAV format
    wav_data = create_test_wav()
    pcm = proc._prepare_audio_data(wav_data)
    assert isinstance(pcm, bytes)
```

### Integration Tests

```python
def test_command_recognition():
    """Test command recognition with real audio"""
    proc = SpeechInputProcessor()
    
    # Load test audio file
    with open('test_audio/start_command.wav', 'rb') as f:
        audio_data = f.read()
    
    result = proc.recognize_speech(audio_data)
    
    assert result['success']
    assert 'start' in result['text'].lower()
    assert result['confidence'] > 0.7
```

## Error Handling

### Graceful Degradation

```python
def recognize_speech(self, audio_data, **kwargs):
    """Recognize speech with comprehensive error handling"""
    
    if not self._ensure_model_loaded():
        return {
            "text": "",
            "success": False,
            "error": self._load_error
        }
    
    try:
        # Recognition logic...
        pass
    except Exception as e:
        # Log error but don't crash
        return {
            "text": "",
            "success": False,
            "error": str(e)
        }
```

### User-Friendly Error Messages

```python
def _get_or_download_model(self):
    """Provide clear instructions if model not found"""
    
    if not os.path.exists(model_path):
        print(f"\n{'='*60}")
        print(f"Vosk model not found: {model_name}")
        print(f"{'='*60}")
        print(f"\nTo use speech recognition, please download a model:")
        print(f"\n1. Visit: https://alphacephei.com/vosk/models")
        print(f"2. Download: {model_name}.zip")
        print(f"3. Extract to: {models_dir}/")
        # ... more instructions
        return None
```

## Best Practices for Integration

### 1. Initialize Once, Use Many Times

```python
class MyProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        # Initialize speech input once
        self.speech_input = SpeechInputProcessor()
    
    def process_multiple_commands(self, audio_list):
        # Reuse same speech_input instance
        results = [
            self.speech_input.recognize_speech(audio)
            for audio in audio_list
        ]
        return results
```

### 2. Handle Asynchronous Recognition

```python
import asyncio

async def recognize_async(processor, audio_data):
    """Async wrapper for recognition"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        processor.recognize_speech,
        audio_data
    )
```

### 3. Implement Timeouts

```python
from concurrent.futures import ThreadPoolExecutor, TimeoutError

def recognize_with_timeout(processor, audio_data, timeout=5.0):
    """Recognition with timeout"""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(processor.recognize_speech, audio_data)
        try:
            result = future.result(timeout=timeout)
            return result
        except TimeoutError:
            return {
                "text": "",
                "success": False,
                "error": "Recognition timeout"
            }
```

## Security Considerations

### 1. Audio Data Validation

```python
def _validate_audio_data(self, audio_data):
    """Validate audio data before processing"""
    
    # Check size limits
    MAX_AUDIO_SIZE = 10 * 1024 * 1024  # 10 MB
    if isinstance(audio_data, bytes) and len(audio_data) > MAX_AUDIO_SIZE:
        raise ValueError("Audio data too large")
    
    # Check format
    if isinstance(audio_data, bytes) and audio_data[:4] == b'RIFF':
        # Validate WAV header
        self._validate_wav_header(audio_data)
```

### 2. Command Sanitization

```python
def sanitize_command(text):
    """Sanitize recognized text for command matching"""
    # Remove potentially harmful characters
    safe_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Limit length
    return safe_text[:1000].strip()
```

## Future Enhancements

Potential improvements for future versions:

1. **Wake Word Detection**: Add "Hey Assistant" style wake word
2. **Speaker Identification**: Recognize different speakers
3. **Real-time Streaming**: Process audio as it arrives
4. **Language Auto-detection**: Automatically detect spoken language
5. **Custom Vocabulary**: Support domain-specific terms
6. **Cloud Fallback**: Use cloud services for complex recognition

## References

- [Vosk API Documentation](https://alphacephei.com/vosk/documentation)
- [Vosk GitHub Repository](https://github.com/alphacep/vosk-api)
- [Speech Recognition Best Practices](https://en.wikipedia.org/wiki/Speech_recognition)
