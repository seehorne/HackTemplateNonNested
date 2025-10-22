# Audio Feedback Processor - Usage Guide

## Overview

The Audio Feedback Processor is a building block that enables other processors to provide non-verbal audio cues for accessibility. This guide shows how to use it in your own processors.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Usage](#basic-usage)
3. [Common Patterns](#common-patterns)
4. [Best Practices](#best-practices)
5. [Examples](#examples)
6. [Troubleshooting](#troubleshooting)

## Getting Started

### Installation

The Audio Feedback Processor is included in the standard WhatsAI environment. No additional dependencies required beyond NumPy (already included).

### Import

```python
from processors.audio_feedback_processor import AudioFeedbackProcessor
```

### Initialization

```python
class MyProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        # Initialize audio feedback as a building block
        self.audio_feedback = AudioFeedbackProcessor()
```

## Basic Usage

### Simple Audio Generation

The simplest way to generate audio is using presets:

```python
def process_frame(self, frame):
    # Your processing logic...
    
    # Generate success audio
    audio = self.audio_feedback.generate_audio_feedback(preset="success")
    
    return frame, {
        "message": "Task completed",
        "audio": audio
    }
```

### Available Presets

- **scanning**: Geiger counter-style clicks for search mode
- **proximity**: Pulsing tone for distance feedback
- **alignment**: Frequency sweep for positioning tasks
- **success**: Positive confirmation tone
- **warning**: Cautionary alert tone
- **error**: Error indication tone

## Common Patterns

### Pattern 1: Proximity-Based Feedback

Perfect for distance detection, object finding, or navigation:

```python
def provide_proximity_feedback(self, distance):
    """
    Generate audio based on distance to target
    
    Args:
        distance: Normalized distance (0.0 = closest, 1.0 = farthest)
    """
    audio = self.audio_feedback.generate_proximity_feedback(
        distance=distance,
        min_distance=0.0,
        max_distance=1.0
    )
    
    return audio
```

**How it works:**
- Closer distances → faster clicks, higher frequency
- Farther distances → slower clicks, lower frequency
- Provides intuitive feedback without words

**Use cases:**
- Object finding (like Object Finder Processor)
- Navigation assistance
- Collision avoidance
- Target approach

### Pattern 2: Alignment/Centering Feedback

Perfect for camera aiming, hand positioning, or object centering:

```python
def provide_alignment_feedback(self, offset_x, offset_y):
    """
    Generate audio based on alignment offset
    
    Args:
        offset_x: Horizontal offset (-1 to 1, 0 = centered)
        offset_y: Vertical offset (-1 to 1, 0 = centered)
    """
    # Calculate total offset
    import math
    total_offset = math.sqrt(offset_x**2 + offset_y**2)
    
    # Generate feedback
    audio = self.audio_feedback.generate_alignment_feedback(
        offset=total_offset,
        aligned_threshold=0.15
    )
    
    return audio
```

**How it works:**
- Aligned (within threshold) → success tone
- Not aligned → sweep tone, intensity based on offset
- Clear distinction between aligned and not aligned

**Use cases:**
- Camera centering (like Camera Aiming Processor)
- Hand positioning
- Object alignment
- Framing assistance

### Pattern 3: State-Based Feedback

Perfect for discrete states or events:

```python
def provide_state_feedback(self, state):
    """
    Generate audio based on processor state
    
    Args:
        state: Current state (e.g., "idle", "active", "success", "error")
    """
    if state == "idle":
        return self.audio_feedback.generate_audio_feedback(preset="scanning")
    elif state == "active":
        return self.audio_feedback.generate_audio_feedback(preset="proximity")
    elif state == "success":
        return self.audio_feedback.generate_audio_feedback(preset="success")
    elif state == "error":
        return self.audio_feedback.generate_audio_feedback(preset="error")
    else:
        # Generic status feedback
        return self.audio_feedback.generate_status_feedback(state)
```

**Use cases:**
- Mode changes
- Event notifications
- Status updates
- Error alerts

### Pattern 4: Custom Audio for Specific Needs

For specialized requirements:

```python
def provide_custom_feedback(self, value):
    """
    Generate custom audio based on specific parameter
    
    Args:
        value: Custom parameter (e.g., confidence score, speed, etc.)
    """
    # Map value to frequency (higher value = higher pitch)
    frequency = 300 + (value * 500)  # 300Hz to 800Hz
    
    # Generate custom audio
    audio = self.audio_feedback.generate_audio_feedback(
        audio_type="sweep",
        frequency=frequency,
        duration=0.2,
        intensity=0.5
    )
    
    return audio
```

## Best Practices

### 1. Choose Appropriate Audio Types

| Use Case | Recommended Type | Why |
|----------|-----------------|-----|
| Continuous scanning | Geiger | Provides ongoing feedback |
| Distance feedback | Proximity helper | Automatic rate/pitch adjustment |
| Alignment tasks | Alignment helper | Clear success indication |
| Discrete events | Beep | Quick, distinct |
| Status changes | Presets | Consistent, recognizable |

### 2. Keep Audio Brief

```python
# Good: Short, responsive
audio = self.audio_feedback.generate_audio_feedback(
    audio_type="beep",
    duration=0.15  # 150ms
)

# Avoid: Too long, delays response
audio = self.audio_feedback.generate_audio_feedback(
    audio_type="tone",
    duration=2.0  # 2 seconds - too long!
)
```

### 3. Avoid Audio Overlap

```python
class MyProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.audio_feedback = AudioFeedbackProcessor()
        self.last_audio_time = 0
        self.min_audio_interval = 0.2  # Minimum 200ms between audio
    
    def process_frame(self, frame):
        import time
        current_time = time.time()
        
        # Only generate audio if enough time has passed
        if current_time - self.last_audio_time >= self.min_audio_interval:
            audio = self.audio_feedback.generate_audio_feedback(preset="proximity")
            self.last_audio_time = current_time
            
            return frame, {"message": "...", "audio": audio}
        else:
            return frame, {"message": "..."}
```

### 4. Use Comfortable Frequencies

```python
# Good: Comfortable range
audio = self.audio_feedback.generate_audio_feedback(
    audio_type="tone",
    frequency=440  # A4 note, comfortable
)

# Avoid: Extreme frequencies can be unpleasant
audio = self.audio_feedback.generate_audio_feedback(
    audio_type="tone",
    frequency=50  # Too low, may not hear
)

audio = self.audio_feedback.generate_audio_feedback(
    audio_type="tone",
    frequency=15000  # Too high, uncomfortable
)
```

### 5. Provide Visual Feedback Too

Always include both audio AND visual/text feedback:

```python
def process_frame(self, frame):
    status = "Object centered"
    
    # Generate audio
    audio = self.audio_feedback.generate_audio_feedback(preset="success")
    
    # Return BOTH message and audio
    return frame, {
        "message": status,  # Visual/text for sighted users
        "audio": audio      # Audio for blind users
    }
```

## Examples

### Example 1: Object Proximity Detector

```python
class ProximityDetector(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.yolo = YOLO("yolo11n.pt")
        self.audio_feedback = AudioFeedbackProcessor()
    
    def process_frame(self, frame):
        height, width = frame.shape[:2]
        results = self.yolo(frame)
        
        if len(results[0].boxes) > 0:
            # Get largest object
            box = results[0].boxes[0]
            bbox = box.xyxy[0].cpu().numpy()
            
            # Calculate size ratio (proxy for distance)
            obj_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            frame_area = width * height
            size_ratio = obj_area / frame_area
            
            # Invert size ratio to distance (larger = closer)
            distance = 1.0 - min(1.0, size_ratio * 2)
            
            # Generate proximity audio
            audio = self.audio_feedback.generate_proximity_feedback(
                distance=distance
            )
            
            message = f"Object detected (distance: {distance:.2f})"
        else:
            # No object - scanning
            audio = self.audio_feedback.generate_audio_feedback(
                preset="scanning"
            )
            message = "Scanning for objects..."
        
        return frame, {"message": message, "audio": audio}
```

### Example 2: Hand Centering Guide

```python
class HandCenteringGuide(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.hand_tracker = HandTrackingProcessor()
        self.audio_feedback = AudioFeedbackProcessor()
    
    def process_frame(self, frame):
        # Get hand data
        _, hand_data = self.hand_tracker._get_hand_data_internal(frame)
        
        if hand_data["hand_count"] > 0:
            hand = hand_data["hands"][0]
            offset_x = hand["offset"]["x"]
            offset_y = hand["offset"]["y"]
            
            # Calculate total offset
            import math
            total_offset = math.sqrt(offset_x**2 + offset_y**2)
            
            # Generate alignment audio
            audio = self.audio_feedback.generate_alignment_feedback(
                offset=total_offset,
                aligned_threshold=0.15
            )
            
            if total_offset < 0.15:
                message = "Hand centered!"
            else:
                directions = []
                if abs(offset_x) > 0.15:
                    directions.append("left" if offset_x > 0 else "right")
                if abs(offset_y) > 0.15:
                    directions.append("up" if offset_y > 0 else "down")
                message = f"Move hand {' and '.join(directions)}"
        else:
            audio = self.audio_feedback.generate_audio_feedback(
                preset="scanning"
            )
            message = "Show your hand"
        
        return frame, {"message": message, "audio": audio}
```

### Example 3: Progress Indicator

```python
class ProgressIndicator(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.audio_feedback = AudioFeedbackProcessor()
        self.progress = 0.0
    
    def update_progress(self, progress):
        """Update progress (0.0 to 1.0)"""
        self.progress = progress
        
        if progress >= 1.0:
            # Complete
            return self.audio_feedback.generate_audio_feedback(
                preset="success"
            )
        elif progress >= 0.8:
            # Almost done - high pitch
            return self.audio_feedback.generate_audio_feedback(
                audio_type="tone",
                frequency=800,
                duration=0.1,
                intensity=0.6
            )
        elif progress >= 0.5:
            # Halfway - medium pitch
            return self.audio_feedback.generate_audio_feedback(
                audio_type="tone",
                frequency=550,
                duration=0.1,
                intensity=0.5
            )
        else:
            # Starting - low pitch
            return self.audio_feedback.generate_audio_feedback(
                audio_type="tone",
                frequency=350,
                duration=0.1,
                intensity=0.4
            )
```

## Troubleshooting

### Audio Not Playing

**Problem**: Generated audio doesn't play in client.

**Solutions**:
1. Verify audio is included in return payload:
   ```python
   return frame, {"message": "...", "audio": audio}
   ```
2. Check audio_data is not empty
3. Verify base64 encoding is correct

### Audio Too Frequent

**Problem**: Audio overlaps or is overwhelming.

**Solution**: Add rate limiting:
```python
import time

class MyProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.audio_feedback = AudioFeedbackProcessor()
        self.last_audio = 0
    
    def generate_audio_with_limit(self, min_interval=0.2):
        current = time.time()
        if current - self.last_audio >= min_interval:
            audio = self.audio_feedback.generate_audio_feedback(...)
            self.last_audio = current
            return audio
        return None
```

### Audio Doesn't Match Expectation

**Problem**: Generated audio doesn't sound as expected.

**Solutions**:
1. Check parameters are in valid ranges
2. Verify preset names are correct
3. Test with simple preset first:
   ```python
   audio = self.audio_feedback.generate_audio_feedback(preset="success")
   ```

### Performance Issues

**Problem**: Audio generation is slow.

**Solutions**:
1. Keep durations short (< 0.5s typically)
2. Avoid generating audio every frame
3. Cache audio if same audio used repeatedly:
   ```python
   class MyProcessor(BaseProcessor):
       def __init__(self):
           super().__init__()
           self.audio_feedback = AudioFeedbackProcessor()
           # Cache common audio
           self.success_audio = self.audio_feedback.generate_audio_feedback(
               preset="success"
           )
   ```

## See Also

- [Audio Feedback Quick Reference](audio_feedback_quick_reference.md) - Quick API reference
- [Full Documentation](AUDIO_FEEDBACK_PROCESSOR.md) - Complete technical documentation
- [Example Processor](../processors/audio_feedback_example_processor.py) - Working example
- [Camera Aiming Processor](../processors/camera_aiming_processor.py) - Real-world usage
- [Object Finder Processor](../processors/object_finder_processor.py) - Advanced integration
