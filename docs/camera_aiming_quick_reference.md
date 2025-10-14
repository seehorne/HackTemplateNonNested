# Camera Aiming Processor - Quick Reference

## Basic Info
- **Processor ID**: 13
- **Port**: 8014
- **Purpose**: Help blind/low vision users center objects in camera frame
- **Dependencies**: None (CPU-only)

## Quick Start

### Standalone Use
1. Select processor ID 13 in web client
2. Point camera at object
3. Listen to audio guidance
4. Adjust camera position based on directions
5. Wait for "Ready to capture!" message

## Return Values

### Status Types
- `perfect` - Centered and sized perfectly ✅
- `centered` - Centered but needs size adjustment
- `adjusting` - Needs repositioning
- `too_close` - Move camera back
- `too_far` - Move camera forward
- `no_object` - Keep scanning

### Example Response
```python
{
    "status": "perfect",
    "message": "person centered and sized perfectly. Ready to capture!",
    "audio_cue": "perfect",
    "centered": True,
    "size_ok": True,
    "object_name": "person",
    "confidence": 0.89,
    "offset_x": 0.02,
    "offset_y": -0.03,
    "size_ratio": 0.45
}
```

## Building Block Usage

### Import and Initialize
```python
from processors.camera_aiming_processor import CameraAimingProcessor

# Basic initialization
aiming = CameraAimingProcessor()

# Target specific object
aiming = CameraAimingProcessor(target_class="person")

# Multiple targets
aiming = CameraAimingProcessor(target_class=["person", "cup", "bottle"])
```

### Get Guidance
```python
# Get aiming guidance for any frame
guidance = aiming.get_aiming_guidance(frame)

# Check if centered
if guidance['centered'] and guidance['size_ok']:
    # Object is perfectly positioned
    do_something(frame)
```

### Process with Visual Output
```python
# Get both visual output and guidance
output_frame, guidance = aiming.process_frame(frame)
```

## Configuration

### Thresholds (can be modified)
```python
CENTER_THRESHOLD = 0.15      # 15% from center is "centered"
SIZE_OPTIMAL_MIN = 0.30      # 30% of frame is minimum optimal
SIZE_OPTIMAL_MAX = 0.70      # 70% of frame is maximum optimal
```

### Constructor Parameters
```python
CameraAimingProcessor(
    model_path="./models/yolo11n-seg.pt",  # YOLO model
    target_class=None,                      # Target object type(s)
    confidence_threshold=0.5                # Detection confidence
)
```

## Integration Patterns

### Pattern 1: Two-Stage Processing
```python
class MyProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.aiming = CameraAimingProcessor()
        self.ready = False
    
    def process_frame(self, frame):
        if not self.ready:
            guidance = self.aiming.get_aiming_guidance(frame)
            if guidance['status'] == 'perfect':
                self.ready = True
            return frame, guidance['message']
        else:
            return self.do_my_processing(frame)
```

### Pattern 2: Continuous Monitoring
```python
def process_frame(self, frame):
    guidance = self.aiming.get_aiming_guidance(frame)
    
    # Do custom processing
    result = self.my_processing(frame)
    
    # Add aiming info to result
    result['aiming_status'] = guidance['status']
    result['centered'] = guidance['centered']
    
    return frame, result
```

### Pattern 3: Conditional Processing
```python
def process_frame(self, frame):
    guidance = self.aiming.get_aiming_guidance(frame)
    
    if guidance['centered']:
        # Only process when object is centered
        return self.process_centered_object(frame)
    else:
        # Guide user to center object
        output, _ = self.aiming.process_frame(frame)
        return output, guidance['message']
```

## Audio Cues

### Cue Types
- `perfect` - Ready to capture
- `centered` - Object centered
- `move_left` - Move camera left
- `move_right` - Move camera right
- `move_up` - Move camera up
- `move_down` - Move camera down
- `move_left_up` - Move camera left and up
- `move_right_down` - Move camera right and down
- (etc. for all direction combinations)
- `move_back` - Too close, move away
- `move_forward` - Too far, move closer
- `scanning` - Looking for object

## Tips

### For Best Results
- Ensure good lighting
- Move camera slowly and steadily
- Wait for confirmation before taking action
- Use consistent object types for targeting

### Common Issues
- **No object detected**: Pan camera slowly
- **Wrong object targeted**: Specify target_class
- **Jittery guidance**: Reduce camera movement
- **Slow detection**: Check CPU usage

## See Also
- Full documentation: [docs/camera_aiming_usage.md](camera_aiming_usage.md)
- Example implementation: [processors/photo_capture_example_processor.py](../processors/photo_capture_example_processor.py)
- Main README: [../README.md](../README.md)
