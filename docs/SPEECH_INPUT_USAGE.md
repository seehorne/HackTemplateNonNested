# Speech Input Processor - Usage Guide

This guide provides practical examples and patterns for integrating the Speech Input Processor into your applications.

## Table of Contents

1. [Basic Setup](#basic-setup)
2. [Simple Voice Commands](#simple-voice-commands)
3. [Voice-Controlled Navigation](#voice-controlled-navigation)
4. [Voice Dictation](#voice-dictation)
5. [Combining with Audio Feedback](#combining-with-audio-feedback)
6. [Multimodal Input (Voice + Hand Tracking)](#multimodal-input)
7. [Real-Time Streaming](#real-time-streaming)
8. [Advanced Command Parsing](#advanced-command-parsing)

---

## Basic Setup

### Installing Vosk Model

Before using the processor, download a speech recognition model:

```bash
# Navigate to models directory
cd models
mkdir -p vosk
cd vosk

# Download small English model (recommended for testing)
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip

# Or download large model for better accuracy
wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip
unzip vosk-model-en-us-0.22.zip
```

### Basic Import and Initialization

```python
from processors.speech_input_processor import SpeechInputProcessor
from processors.base_processor import BaseProcessor

class MyVoiceProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        
        # Initialize speech input building block
        self.speech_input = SpeechInputProcessor(
            language="en-us",
            model_size="small"  # or "large"
        )
```

---

## Simple Voice Commands

### Example: Basic Command Recognition

```python
from processors.speech_input_processor import SpeechInputProcessor
from processors.base_processor import BaseProcessor
import numpy as np

class VoiceCommandProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.speech_input = SpeechInputProcessor()
        self.state = "idle"
    
    def process_audio_command(self, audio_data):
        """Process voice command from audio data"""
        result = self.speech_input.recognize_speech(audio_data)
        
        if not result['success']:
            return {
                "status": "error",
                "message": "Could not recognize speech"
            }
        
        text = result['text'].lower()
        
        # Simple command matching
        if 'start' in text:
            self.state = "running"
            return {"command": "start", "state": self.state}
        elif 'stop' in text:
            self.state = "stopped"
            return {"command": "stop", "state": self.state}
        elif 'pause' in text:
            self.state = "paused"
            return {"command": "pause", "state": self.state}
        else:
            return {
                "status": "unknown",
                "text": text,
                "message": "Command not recognized"
            }
```

### Example: Using Command Handler Utility

```python
class VoiceControlledApp(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.speech_input = SpeechInputProcessor()
        self.is_running = False
        
        # Define command callbacks
        commands = {
            "start": self.start_app,
            "stop": self.stop_app,
            "reset": self.reset_app,
            "help": self.show_help
        }
        
        # Create command handler
        self.command_handler = self.speech_input.create_voice_command_handler(commands)
    
    def start_app(self):
        self.is_running = True
        print("Application started")
    
    def stop_app(self):
        self.is_running = False
        print("Application stopped")
    
    def reset_app(self):
        self.is_running = False
        print("Application reset")
    
    def show_help(self):
        print("Available commands: start, stop, reset, help")
    
    def process_audio(self, audio_data):
        result = self.command_handler(audio_data)
        return result
```

---

## Voice-Controlled Navigation

### Example: Directional Control

```python
class VoiceNavigationProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.speech_input = SpeechInputProcessor()
        self.position = {"x": 0, "y": 0}
        self.speed = 10
    
    def process_navigation_command(self, audio_data):
        """Process directional voice commands"""
        result = self.speech_input.recognize_speech(audio_data)
        
        if not result['success']:
            return self.position
        
        text = result['text'].lower()
        
        # Parse directional commands
        if 'left' in text:
            self.position['x'] -= self.speed
        elif 'right' in text:
            self.position['x'] += self.speed
        
        if 'up' in text:
            self.position['y'] -= self.speed
        elif 'down' in text:
            self.position['y'] += self.speed
        
        # Parse speed modifiers
        if 'fast' in text or 'faster' in text:
            self.speed = min(self.speed + 5, 50)
        elif 'slow' in text or 'slower' in text:
            self.speed = max(self.speed - 5, 1)
        
        # Reset position
        if 'center' in text or 'reset' in text:
            self.position = {"x": 0, "y": 0}
        
        return {
            "position": self.position,
            "speed": self.speed,
            "command": text
        }
```

---

## Voice Dictation

### Example: Text Input via Voice

```python
class VoiceDictationProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.speech_input = SpeechInputProcessor(model_size="large")  # Better for dictation
        self.text_buffer = []
        self.current_document = ""
    
    def process_dictation(self, audio_data):
        """Process voice dictation"""
        result = self.speech_input.recognize_speech(
            audio_data,
            mode=self.speech_input.MODE_DICTATION
        )
        
        if result['success']:
            # Add to buffer
            self.text_buffer.append(result['text'])
            
            # Update document
            self.current_document = " ".join(self.text_buffer)
            
            return {
                "text": result['text'],
                "document": self.current_document,
                "word_count": len(self.current_document.split())
            }
        
        return {"error": "Recognition failed"}
    
    def clear_document(self):
        """Clear the current document"""
        self.text_buffer = []
        self.current_document = ""
    
    def undo_last(self):
        """Remove last dictated segment"""
        if self.text_buffer:
            self.text_buffer.pop()
            self.current_document = " ".join(self.text_buffer)
```

---

## Combining with Audio Feedback

### Example: Voice Commands with Audio Confirmation

```python
from processors.speech_input_processor import SpeechInputProcessor
from processors.audio_feedback_processor import AudioFeedbackProcessor
from processors.base_processor import BaseProcessor

class VoiceWithFeedbackProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.speech_input = SpeechInputProcessor()
        self.audio_feedback = AudioFeedbackProcessor()
        self.state = "idle"
    
    def process_voice_with_feedback(self, audio_data):
        """Process voice command and provide audio feedback"""
        # Recognize speech
        result = self.speech_input.recognize_speech(audio_data)
        
        response = {}
        
        if result['success']:
            # Command recognized successfully
            text = result['text'].lower()
            
            # Generate success audio
            feedback = self.audio_feedback.generate_audio_feedback(
                preset="success"
            )
            
            # Process command
            if 'start' in text:
                self.state = "running"
            elif 'stop' in text:
                self.state = "stopped"
            
            response = {
                "status": "success",
                "command": text,
                "state": self.state,
                "audio": feedback
            }
        else:
            # Recognition failed
            feedback = self.audio_feedback.generate_audio_feedback(
                preset="error"
            )
            
            response = {
                "status": "error",
                "message": "Command not recognized",
                "audio": feedback
            }
        
        return response
```

### Example: Progressive Audio Feedback During Recognition

```python
class ProgressiveFeedbackProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.speech_input = SpeechInputProcessor()
        self.audio_feedback = AudioFeedbackProcessor()
    
    def process_with_confidence_feedback(self, audio_data):
        """Provide audio feedback based on recognition confidence"""
        result = self.speech_input.recognize_speech(audio_data)
        
        if not result['success']:
            # No speech detected - scanning feedback
            feedback = self.audio_feedback.generate_audio_feedback(
                preset="scanning"
            )
            return {"status": "listening", "audio": feedback}
        
        confidence = result['confidence']
        
        # Generate feedback based on confidence
        if confidence > 0.9:
            # High confidence - success
            feedback = self.audio_feedback.generate_audio_feedback(
                preset="success"
            )
        elif confidence > 0.7:
            # Medium confidence - proximity
            feedback = self.audio_feedback.generate_audio_feedback(
                preset="proximity"
            )
        else:
            # Low confidence - warning
            feedback = self.audio_feedback.generate_audio_feedback(
                preset="warning"
            )
        
        return {
            "text": result['text'],
            "confidence": confidence,
            "audio": feedback
        }
```

---

## Multimodal Input

### Example: Voice + Hand Tracking

```python
from processors.speech_input_processor import SpeechInputProcessor
from processors.hand_tracking_processor import HandTrackingProcessor
from processors.base_processor import BaseProcessor

class MultimodalInputProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.speech_input = SpeechInputProcessor()
        self.hand_tracker = HandTrackingProcessor()
    
    def process_multimodal(self, frame, audio_data):
        """Combine voice commands with hand gestures"""
        # Get hand tracking data
        hand_data = self.hand_tracker.get_hand_tracking_data(frame)
        
        # Get voice command
        speech_result = self.speech_input.recognize_speech(audio_data)
        
        # Combine inputs
        result = {
            "hand_detected": hand_data['status'] != 'no_hands',
            "voice_detected": speech_result['success']
        }
        
        if hand_data['status'] != 'no_hands':
            hand = hand_data['hands'][0]
            result['hand_location'] = hand['location']
            result['hand_distance'] = hand['distance']
        
        if speech_result['success']:
            text = speech_result['text'].lower()
            result['voice_command'] = text
            
            # Example: "point here" - use hand position with voice confirmation
            if 'point' in text or 'here' in text:
                if result['hand_detected']:
                    result['action'] = 'selection'
                    result['selection_location'] = result['hand_location']
            
            # Example: "grab" - use hand gesture with voice trigger
            elif 'grab' in text or 'take' in text:
                if result['hand_detected']:
                    result['action'] = 'grab'
        
        return result
```

---

## Real-Time Streaming

### Example: Continuous Speech Recognition

```python
class StreamingRecognitionProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.speech_input = SpeechInputProcessor()
        self.audio_buffer = []
        self.partial_results = []
    
    def add_audio_chunk(self, audio_chunk):
        """Add audio chunk to buffer"""
        self.audio_buffer.append(audio_chunk)
    
    def process_streaming(self, get_partial=True):
        """Process accumulated audio chunks"""
        if not self.audio_buffer:
            return {"status": "no_data"}
        
        # Process all buffered chunks
        results = self.speech_input.recognize_speech_streaming(
            self.audio_buffer,
            get_partial=get_partial
        )
        
        # Clear buffer
        self.audio_buffer = []
        
        # Extract final results
        final_results = [r for r in results if not r.get('partial', False)]
        partial_results = [r for r in results if r.get('partial', False)]
        
        return {
            "final": final_results,
            "partial": partial_results,
            "total_chunks": len(results)
        }
```

---

## Advanced Command Parsing

### Example: Natural Language Commands

```python
class NaturalLanguageProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.speech_input = SpeechInputProcessor()
    
    def parse_natural_command(self, audio_data):
        """Parse natural language commands"""
        result = self.speech_input.recognize_speech(audio_data)
        
        if not result['success']:
            return {"status": "error"}
        
        text = result['text'].lower()
        
        # Parse complex commands
        command = {
            "action": None,
            "object": None,
            "modifier": None,
            "value": None
        }
        
        # Action verbs
        if 'show' in text or 'display' in text:
            command['action'] = 'show'
        elif 'hide' in text or 'close' in text:
            command['action'] = 'hide'
        elif 'increase' in text or 'raise' in text:
            command['action'] = 'increase'
        elif 'decrease' in text or 'lower' in text:
            command['action'] = 'decrease'
        
        # Objects
        if 'volume' in text:
            command['object'] = 'volume'
        elif 'brightness' in text:
            command['object'] = 'brightness'
        elif 'menu' in text:
            command['object'] = 'menu'
        
        # Modifiers
        if 'maximum' in text or 'max' in text:
            command['modifier'] = 'max'
        elif 'minimum' in text or 'min' in text:
            command['modifier'] = 'min'
        
        # Extract numeric values
        import re
        numbers = re.findall(r'\d+', text)
        if numbers:
            command['value'] = int(numbers[0])
        
        return {
            "raw_text": text,
            "parsed": command,
            "confidence": result['confidence']
        }
```

---

## Best Practices

### 1. Error Handling

Always check for success before using recognition results:

```python
result = self.speech_input.recognize_speech(audio_data)

if not result['success']:
    # Handle failure
    error_message = result.get('error', 'Unknown error')
    return {"status": "error", "message": error_message}

# Use result safely
text = result['text']
```

### 2. Confidence Thresholds

Filter low-confidence results:

```python
MIN_CONFIDENCE = 0.7

result = self.speech_input.recognize_speech(audio_data)

if result['success'] and result['confidence'] >= MIN_CONFIDENCE:
    # Process high-confidence result
    process_command(result['text'])
else:
    # Request repeat or provide feedback
    return {"status": "uncertain", "message": "Please repeat command"}
```

### 3. Text Normalization

Always normalize text for command matching:

```python
text = result['text'].lower().strip()

# Remove common filler words
text = text.replace('please', '').replace('um', '').strip()

# Match commands
if text in ['start', 'begin', 'go']:
    start_process()
```

### 4. User Feedback

Provide clear feedback for voice interactions:

```python
result = self.speech_input.recognize_speech(audio_data)

feedback = {
    "recognized": result['text'],
    "confidence": f"{result['confidence']*100:.0f}%",
    "status": "Command executed" if result['success'] else "Not recognized"
}

return feedback
```

---

## See Also

- [Speech Input Processor Documentation](SPEECH_INPUT_PROCESSOR.md)
- [Speech Input Quick Reference](speech_input_quick_reference.md)
- [Example Processor](../processors/speech_command_example_processor.py)
