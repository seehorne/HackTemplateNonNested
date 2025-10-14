# Camera Aiming Processor - Usage Guide

The Camera Aiming Processor (ID: 13) is a building block processor designed to help users who are blind or low vision aim their camera to properly center objects for photography or interaction.

## Overview

This processor provides **non-visual audio cues** to guide camera positioning:
- Directional guidance (move left/right/up/down)
- Centering confirmation (object is properly centered)
- Distance feedback (move closer/farther)
- Size optimization (object fills frame appropriately)

## Features

### 🎯 Directional Guidance
The processor tells you which direction to move your camera to center the target object:
- "Move camera left to center [object]"
- "Move camera right and up to center [object]"
- And more directional combinations

### ✅ Centering Confirmation
When your object is properly centered:
- "Object centered and sized perfectly. Ready to capture!"
- "Object centered. Good alignment!"

### 📏 Distance Feedback
Helps you get the right distance from the object:
- "Object centered but too close. Move camera back."
- "Object centered but too far. Move camera closer."

### 📐 Size Optimization
The processor considers object size:
- **Too small** (< 15% of frame): Object is too far
- **Optimal** (30-70% of frame): Perfect size for capture
- **Too large** (> 85% of frame): Object is too close

## Basic Usage

### As a Standalone Processor

1. Enable the Camera Aiming Processor (ID: 13) in the web client
2. Point your camera at any object you want to center
3. Listen to the audio guidance
4. Adjust your camera position based on the directions
5. When you hear "Ready to capture!", the object is perfectly centered

### API Response Format

The processor returns a structured guidance dictionary:

```json
{
  "status": "centered|adjusting|perfect|too_close|too_far|no_object",
  "message": "Human-readable guidance message",
  "audio_cue": "Audio cue identifier for TTS or sound effects",
  "centered": true/false,
  "size_ok": true/false,
  "object_name": "person|car|cup|etc.",
  "confidence": 0.85,
  "offset_x": 0.05,
  "offset_y": -0.10,
  "size_ratio": 0.45,
  "distance_from_center": 0.11,
  "too_close": false,
  "too_far": false
}
```

### Status Codes

- `perfect`: Object is centered and sized optimally - ready to capture
- `centered`: Object is centered but size may need adjustment
- `adjusting`: Object detected but needs repositioning
- `too_close`: Object is centered but camera is too close
- `too_far`: Object is centered but camera is too far
- `no_object`: No object detected in frame
- `scanning`: Actively looking for objects (used during no_object state)

## Using as a Building Block

Other processors can import and use the camera aiming functionality:

### Example: Simple Integration

```python
from processors.camera_aiming_processor import CameraAimingProcessor

class MyCustomProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        # Initialize camera aiming as a component
        self.aiming_helper = CameraAimingProcessor()
    
    def process_frame(self, frame):
        # Get aiming guidance
        guidance = self.aiming_helper.get_aiming_guidance(frame)
        
        if guidance['centered'] and guidance['size_ok']:
            # Object is properly centered, do your custom processing
            return self.do_custom_processing(frame)
        else:
            # Return guidance to help user center the object
            return frame, guidance['message']
```

### Example: Target-Specific Class

```python
# Only guide toward specific object types
guidance = self.aiming_helper.get_aiming_guidance(frame, target_class="cup")
```

### Example: Multi-Stage Processing

```python
class PhotoCaptureProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.aiming_helper = CameraAimingProcessor(target_class="person")
        self.capture_mode = False
    
    def process_frame(self, frame):
        if not self.capture_mode:
            # First stage: Help user aim camera
            guidance = self.aiming_helper.get_aiming_guidance(frame)
            
            if guidance['status'] == 'perfect':
                self.capture_mode = True
                return frame, "Perfect! Capturing in 3 seconds..."
            else:
                return frame, guidance['message']
        else:
            # Second stage: Capture and process photo
            return self.capture_photo(frame)
```

## Configuration Options

When instantiating the processor, you can configure:

```python
processor = CameraAimingProcessor(
    model_path="./models/yolo11n-seg.pt",  # Path to YOLO model
    target_class="person",                   # Specific class to aim at
    confidence_threshold=0.5                 # Minimum detection confidence
)
```

### Target Class Options

You can specify single or multiple target classes:

```python
# Single class
processor = CameraAimingProcessor(target_class="person")

# Multiple classes
processor = CameraAimingProcessor(target_class=["person", "cup", "bottle"])

# Any object (default)
processor = CameraAimingProcessor()  # Will aim at largest/closest object
```

## Accessibility Features

### Non-Visual Design
- All guidance is provided through text that can be read by screen readers or TTS
- Audio cues are structured for easy sonification
- No reliance on visual indicators for core functionality

### Visual Overlay (Optional)
For sighted users or debugging, the processor provides visual feedback:
- Green bounding box when object is centered
- Orange bounding box when adjustment is needed
- Crosshairs showing frame center
- Line connecting object center to frame center

### CPU-Only Compatible
- Uses YOLO11 which works efficiently on CPU
- No GPU required
- Optimized for real-time performance on modest hardware

## Technical Details

### Centering Thresholds

```python
CENTER_THRESHOLD = 0.15      # Object center within 15% of frame center
SIZE_MIN_THRESHOLD = 0.15    # Minimum 15% of frame
SIZE_MAX_THRESHOLD = 0.85    # Maximum 85% of frame
SIZE_OPTIMAL_MIN = 0.30      # Optimal range: 30-70%
SIZE_OPTIMAL_MAX = 0.70
```

### Object Selection Priority

When multiple objects are detected:
1. If `target_class` is specified: Selects largest object of that class
2. If no target class: Selects object closest to center

### Coordinate System

- `offset_x`: Horizontal offset (-1 = far left, 0 = centered, +1 = far right)
- `offset_y`: Vertical offset (-1 = top, 0 = centered, +1 = bottom)
- Note: Guidance directions are camera-relative (opposite of object position)
  - Object to right → "Move camera left"
  - Object above → "Move camera up"

## Future Enhancements

This building block is designed to be extended. Potential enhancements:

1. **Audio Feedback Integration**: Connect audio cues to actual sound effects
2. **Haptic Feedback**: Add vibration patterns for mobile devices
3. **Multiple Object Tracking**: Guide toward groups of objects
4. **Distance Estimation**: Use depth information for precise distance
5. **Timed Capture**: Auto-capture when centered for specified duration
6. **Custom Centering Zones**: Define specific areas for object placement

## Examples of Processors That Could Use This

1. **Photo Capture Processor**: Help user frame portraits
2. **Object Recognition Processor**: Center objects before identifying them
3. **Document Scanner Processor**: Align documents for scanning
4. **QR Code Reader**: Position QR codes for scanning
5. **Product Recognition Processor**: Center products for identification
6. **Navigation Assistant**: Center doorways or landmarks

## Troubleshooting

### "No object detected" Message
- Ensure there's sufficient lighting
- Point camera toward a recognizable object
- Move camera slowly in a scanning pattern

### Object Not Centering Properly
- Check that the right object is being detected
- Verify object is at reasonable distance (not too close/far)
- Ensure object is not occluded

### Performance Issues
- Reduce frame rate if processing is slow
- Ensure YOLO model is properly loaded
- Check system resources (CPU usage)

## Support

For issues or questions about the Camera Aiming Processor:
- Check the main README.md for system setup
- Review processor_config.json for configuration
- Examine the source code in processors/camera_aiming_processor.py
