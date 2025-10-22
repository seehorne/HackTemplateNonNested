# Audio Feedback Processor - Quick Reference

## Quick Start

```python
from processors.audio_feedback_processor import AudioFeedbackProcessor

# Initialize
audio_proc = AudioFeedbackProcessor()

# Generate audio
audio = audio_proc.generate_audio_feedback(preset="success")
# Returns: {"audio_data": "base64_wav", "duration": 0.2, ...}
```

## Common Use Cases

### 1. Proximity/Distance Feedback (Geiger Counter Style)
```python
# Closer = faster clicks, higher frequency
audio = audio_proc.generate_proximity_feedback(
    distance=0.3,        # 0.0 = closest, 1.0 = farthest
    min_distance=0.0,
    max_distance=1.0
)
```

### 2. Alignment/Centering Feedback
```python
# Success tone when aligned, sweep when not
audio = audio_proc.generate_alignment_feedback(
    offset=0.05,             # -1.0 to 1.0 (0 = centered)
    aligned_threshold=0.1    # Within ±0.1 = aligned
)
```

### 3. Status Events
```python
# Auto-detects status type from string
success_audio = audio_proc.generate_status_feedback("success")
warning_audio = audio_proc.generate_status_feedback("warning")
error_audio = audio_proc.generate_status_feedback("error")
```

## Presets Reference

| Preset | Type | Use Case |
|--------|------|----------|
| `scanning` | Geiger clicks | Search/scan mode |
| `proximity` | Pulsing tone | Distance feedback |
| `alignment` | Sweep | Positioning/centering |
| `success` | Double beep | Task completed |
| `warning` | Double beep | Caution alert |
| `error` | Low tone | Error state |

```python
# Use any preset
audio = audio_proc.generate_audio_feedback(preset="scanning")
```

## Audio Types

### Beep
```python
audio = audio_proc.generate_audio_feedback(
    audio_type="beep",
    frequency=440,
    duration=0.15,
    intensity=0.6
)
```

### Tone
```python
audio = audio_proc.generate_audio_feedback(
    audio_type="tone",
    frequency=880,
    duration=0.5,
    intensity=0.4
)
```

### Sweep (Frequency Chirp)
```python
audio = audio_proc.generate_audio_feedback(
    audio_type="sweep",
    frequency=300,      # Sweeps from 300Hz to 600Hz
    duration=0.3,
    intensity=0.5
)
```

### Pulse (Amplitude Modulated)
```python
audio = audio_proc.generate_audio_feedback(
    audio_type="pulse",
    frequency=440,
    duration=0.6,
    intensity=0.5
)
```

### Geiger (Random Clicks)
```python
audio = audio_proc.generate_audio_feedback(
    audio_type="geiger",
    duration=0.2,
    intensity=0.7       # 0.1 = slow, 1.0 = rapid
)
```

## Patterns

Create complex audio with pauses:

```python
# Three beeps: beep(0.1s) pause(0.05s) beep(0.1s) pause(0.05s) beep(0.1s)
audio = audio_proc.generate_audio_feedback(
    audio_type="beep",
    frequency=440,
    duration=0.1,
    intensity=0.6,
    pattern=[0.1, 0.05, 0.1, 0.05, 0.1]
)
```

## Integration Example

```python
class MyProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.audio_feedback = AudioFeedbackProcessor()
    
    def process_frame(self, frame):
        # Your processing...
        distance = calculate_distance()
        
        # Generate audio
        audio = self.audio_feedback.generate_proximity_feedback(
            distance=distance,
            min_distance=0.0,
            max_distance=2.0
        )
        
        return frame, {
            "message": f"Distance: {distance:.2f}m",
            "audio": audio
        }
```

## Parameter Ranges

| Parameter | Min | Max | Default | Notes |
|-----------|-----|-----|---------|-------|
| frequency | 20 Hz | 20000 Hz | 440 Hz | Human hearing range |
| duration | 0.01 s | 10.0 s | 0.2 s | Keep short for responsiveness |
| intensity | 0.0 | 1.0 | 0.5 | Volume level |

## Output Format

```python
{
    "audio_data": "UklGRgxF...",      # Base64 encoded WAV
    "duration": 0.2,                  # Actual duration in seconds
    "sample_rate": 44100,             # 44.1kHz (CD quality)
    "description": "Beep at 440Hz",   # Human-readable description
    "format": "wav"                   # Always WAV
}
```

## Tips

1. **Keep It Short**: 0.1-0.5s is usually ideal
2. **Comfortable Frequency**: 200-2000 Hz works well
3. **Moderate Volume**: Start with intensity=0.5
4. **Test with Users**: Get feedback from target audience
5. **Avoid Overlap**: Don't generate too frequently
6. **Use Presets**: Maintains consistency

## See Also

- [Full Documentation](AUDIO_FEEDBACK_PROCESSOR.md)
- [Example Processor](../processors/audio_feedback_example_processor.py)
- [Camera Aiming Processor](../processors/camera_aiming_processor.py) - Example usage
