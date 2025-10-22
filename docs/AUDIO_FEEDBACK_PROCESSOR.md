# Audio Feedback Building Block Processor

## Overview

The Audio Feedback Processor is a reusable building block that generates non-verbal audio cues for accessibility and user feedback. It's designed to be integrated into other processors that need audio feedback capabilities.

## Features

- **CPU-Only**: Pure NumPy-based audio generation, no GPU required
- **Multiple Audio Types**: Beeps, tones, sweeps, pulses, and Geiger counter-style clicks
- **Configurable Parameters**: Frequency, duration, intensity, and patterns
- **Preset Modes**: Ready-to-use presets for common scenarios
- **Building Block Design**: Easy integration into other processors
- **WAV Format Output**: Base64-encoded WAV audio compatible with web browsers

## Audio Types

### 1. Beep
Simple beep with smooth envelope (attack/release).
- **Use Case**: Confirmation, button press, discrete events
- **Parameters**: frequency, duration, intensity

### 2. Tone
Sustained pure tone.
- **Use Case**: Sustained states, continuous feedback
- **Parameters**: frequency, duration, intensity

### 3. Sweep
Frequency sweep (chirp) from start to end frequency.
- **Use Case**: Transitions, alignment guidance
- **Parameters**: start/end frequency, duration, intensity

### 4. Pulse
Amplitude-modulated tone (pulsing effect).
- **Use Case**: Attention-getting, periodic events
- **Parameters**: frequency, duration, intensity

### 5. Geiger Counter
Random clicks with rate proportional to intensity.
- **Use Case**: Proximity sensing, scanning, search
- **Parameters**: duration, intensity (controls click rate)

## Presets

Ready-to-use presets for common scenarios:

- **`scanning`**: Geiger-style clicks for search/scan mode
- **`proximity`**: Pulsing tone for distance feedback
- **`alignment`**: Sweep for alignment/positioning
- **`success`**: Double beep for successful completion
- **`warning`**: Double beep pattern for warnings
- **`error`**: Low tone for errors

## Usage

### As a Standalone Processor

```python
from processors.audio_feedback_processor import AudioFeedbackProcessor

# Initialize
audio_proc = AudioFeedbackProcessor()

# Generate simple beep
audio_data = audio_proc.generate_audio_feedback(
    audio_type="beep",
    frequency=440.0,
    duration=0.2,
    intensity=0.5
)

# Generate using preset
audio_data = audio_proc.generate_audio_feedback(
    preset="success"
)
```

### As a Building Block in Other Processors

```python
from processors.audio_feedback_processor import AudioFeedbackProcessor

class MyProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        # Initialize audio feedback as a building block
        self.audio_feedback = AudioFeedbackProcessor()
    
    def process_frame(self, frame):
        # Your processing logic here...
        
        # Generate proximity feedback
        audio = self.audio_feedback.generate_proximity_feedback(
            distance=0.3,
            min_distance=0.0,
            max_distance=1.0
        )
        
        # Include audio in result
        return frame, {
            "message": "Object nearby",
            "audio": audio
        }
```

### Helper Methods

#### Proximity Feedback
Automatically generates audio based on distance with varying intensity and frequency:

```python
audio = audio_proc.generate_proximity_feedback(
    distance=0.5,        # Current distance (0-1)
    min_distance=0.0,
    max_distance=1.0
)
```

**Behavior**: 
- Closer = faster clicks, higher frequency
- Further = slower clicks, lower frequency

#### Alignment Feedback
Generates audio based on alignment offset:

```python
audio = audio_proc.generate_alignment_feedback(
    offset=0.2,              # Offset from target (-1 to 1)
    aligned_threshold=0.1    # Threshold for "aligned"
)
```

**Behavior**:
- Aligned (within threshold) = success tone
- Not aligned = sweep with intensity based on offset

#### Status Feedback
Generates audio based on status string:

```python
audio = audio_proc.generate_status_feedback("success")
# Also recognizes: warning, error, scanning, complete, fail, etc.
```

## API Reference

### `generate_audio_feedback()`

Main method for generating audio.

**Parameters:**
- `audio_type` (str): Type of audio - "beep", "tone", "sweep", "pulse", "geiger"
- `duration` (float): Duration in seconds (0.01 to 10.0)
- `frequency` (float): Base frequency in Hz (20 to 20000)
- `intensity` (float): Volume/intensity (0.0 to 1.0)
- `pattern` (list, optional): Pattern of [duration1, pause1, duration2, pause2, ...]
- `preset` (str, optional): Preset name (overrides other parameters)

**Returns:**
```python
{
    "audio_data": "base64_encoded_wav_data",
    "duration": 0.2,
    "sample_rate": 44100,
    "description": "Beep at 440Hz for 0.2s",
    "format": "wav"
}
```

## Examples

### Example 1: Simple Beep
```python
audio = audio_proc.generate_audio_feedback(
    audio_type="beep",
    frequency=880,
    duration=0.15,
    intensity=0.6
)
```

### Example 2: Geiger Counter for Scanning
```python
audio = audio_proc.generate_audio_feedback(
    audio_type="geiger",
    duration=0.5,
    intensity=0.3  # Slow clicks
)
```

### Example 3: Frequency Sweep
```python
audio = audio_proc.generate_audio_feedback(
    audio_type="sweep",
    frequency=300,  # Will sweep from 300Hz to 600Hz
    duration=0.4,
    intensity=0.5
)
```

### Example 4: Patterned Beeps
```python
audio = audio_proc.generate_audio_feedback(
    audio_type="beep",
    frequency=440,
    duration=0.5,
    intensity=0.6,
    pattern=[0.1, 0.05, 0.1, 0.05, 0.1]  # Three short beeps
)
```

### Example 5: Using Presets
```python
# Success sound
success_audio = audio_proc.generate_audio_feedback(preset="success")

# Warning sound
warning_audio = audio_proc.generate_audio_feedback(preset="warning")

# Scanning sound
scan_audio = audio_proc.generate_audio_feedback(preset="scanning")
```

## Integration Examples

### Object Detection with Proximity Audio
```python
class ObjectProximityProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.yolo = YOLO("yolo11n.pt")
        self.audio_feedback = AudioFeedbackProcessor()
    
    def process_frame(self, frame):
        results = self.yolo(frame)
        
        if len(results[0].boxes) > 0:
            # Get closest object
            box = results[0].boxes[0]
            area_ratio = calculate_area_ratio(box)
            
            # Generate proximity audio
            audio = self.audio_feedback.generate_proximity_feedback(
                distance=1.0 - area_ratio,
                min_distance=0.0,
                max_distance=1.0
            )
            
            return frame, {
                "message": "Object detected",
                "audio": audio
            }
        else:
            # No object - scanning audio
            audio = self.audio_feedback.generate_audio_feedback(
                preset="scanning"
            )
            return frame, {"message": "Scanning", "audio": audio}
```

### Hand Tracking with Directional Audio
```python
class HandGuidanceProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.hand_tracker = HandTrackingProcessor()
        self.audio_feedback = AudioFeedbackProcessor()
    
    def process_frame(self, frame):
        hand_data = self.hand_tracker.get_hand_tracking_data(frame)
        
        if hand_data["hand_count"] > 0:
            hand = hand_data["hands"][0]
            offset_x = hand["offset"]["x"]
            
            # Generate alignment audio based on centering
            audio = self.audio_feedback.generate_alignment_feedback(
                offset=offset_x,
                aligned_threshold=0.15
            )
            
            return frame, {
                "message": f"Hand {hand['location']}",
                "audio": audio
            }
        
        return frame, {"message": "No hand detected"}
```

## Technical Details

- **Sample Rate**: 44100 Hz (CD quality)
- **Format**: 16-bit PCM WAV
- **Channels**: Mono (1 channel)
- **Encoding**: Base64 for transmission
- **Audio Generation**: Pure NumPy (no external audio libraries required)
- **CPU Only**: No GPU dependencies

## Processor Configuration

Add to `processor_config.json`:

```json
{
  "16": {
    "host": "127.0.0.1",
    "port": 8017,
    "name": "audio_feedback_processor",
    "conda_env": "whatsai",
    "dependencies": [],
    "expects_input": "image",
    "description": "Building block processor for generating non-verbal audio feedback",
    "enabled": true
  }
}
```

## Future Enhancements

Potential additions for future versions:

1. **Spatial Audio**: Pan left/right for directional cues
2. **Multiple Tones**: Chords and harmonics
3. **Voice Synthesis**: Optional text-to-speech integration
4. **Custom Waveforms**: Square, triangle, sawtooth waves
5. **Advanced Patterns**: Rhythmic patterns, acceleration/deceleration
6. **Audio Effects**: Reverb, echo, filters

## Best Practices

1. **Keep Duration Short**: Audio feedback should be brief (0.1-0.5s typically)
2. **Avoid Frequency Extremes**: Stay in comfortable range (200-2000 Hz)
3. **Moderate Intensity**: Start around 0.5 and adjust based on use case
4. **Use Presets**: Leverage presets for consistency across processors
5. **Test with Users**: Get feedback from blind/low-vision users on audio design
6. **Consider Context**: Match audio type to the feedback purpose
7. **Avoid Overlap**: Don't generate audio too frequently (can be overwhelming)

## Accessibility Considerations

- **Frequency Range**: Designed for human hearing comfort (not too high/low)
- **Volume Control**: Intensity parameter allows user control
- **Non-Overlapping**: Short durations prevent audio overlap
- **Distinct Patterns**: Each preset has unique characteristics for easy recognition
- **Progressive Feedback**: Geiger/proximity modes provide gradual information
- **Status Clarity**: Success/warning/error sounds are easily distinguishable

## See Also

- [Camera Aiming Processor](../processors/camera_aiming_processor.py) - Example using directional feedback
- [Hand Tracking Processor](../processors/hand_tracking_processor.py) - Example using spatial feedback
- [Object Finder Processor](../processors/object_finder_processor.py) - Example combining multiple building blocks
- [Audio Feedback Example](../processors/audio_feedback_example_processor.py) - Complete integration example
