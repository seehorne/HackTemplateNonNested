# Object Finder Processor

## Overview

The Object Finder Processor helps blind and low vision users locate and reach objects using their hands through non-visual audio cues. It combines hand tracking and object detection to provide real-time guidance for object retrieval.

## Key Features

- **Hand-to-Object Guidance**: Provides directional audio cues to guide the user's hand to detected objects
- **Spatial Audio**: Uses stereo panning to indicate object direction (left/right)
- **Distance Feedback**: Tells the user how far their hand is from the target object
- **Reach Confirmation**: Announces when the hand reaches the target object
- **Object Identification**: Identifies what object the user is reaching for
- **Building Block Integration**: Demonstrates how to combine multiple building block processors
- **CPU-Only**: Works on CPU without requiring GPU

## How It Works

1. **Object Detection**: Uses YOLO11 to detect objects in the camera view
2. **Hand Tracking**: Uses MediaPipe to track the user's hand position
3. **Distance Calculation**: Calculates the distance and direction from hand to closest object
4. **Audio Guidance**: Provides real-time directional audio cues
5. **Reach Confirmation**: Announces when the object is reached

## Usage

### As a Standalone Processor

```python
from processors.object_finder_processor import ObjectFinderProcessor

processor = ObjectFinderProcessor()
output_frame, result = processor.process_frame(frame)

print(result['message'])  # "Move hand right. Cup is medium distance."
print(result['pan'])      # 0.5 (spatial audio position)
```

### Configuration Options

```python
processor = ObjectFinderProcessor(
    model_path="./models/yolo11n-seg.pt",  # YOLO model path
    target_object_class=None,              # Specific object to find (None = any)
    guidance_threshold=50                  # Pixels to consider "reached"
)
```

## Audio Feedback Messages

### No Hand Detected
- "Show your hand to start finding objects."

### No Objects Detected
- "No objects detected. Move camera to scan the area."

### Guidance Messages
- "Move hand right. Cup is medium distance."
- "Move hand left and down. Book is far away."
- "Almost there! Phone is very close."

### Reached Object
- "Object reached! Cup is right there."

## Response Format

The processor returns a dictionary with:

```python
{
    "message": "Move hand right. Cup is medium distance.",
    "pan": 0.5,                    # -1 (left) to 1 (right) for spatial audio
    "hand_detected": True,          # Whether a hand is detected
    "objects_detected": True,       # Whether objects are detected
    "object_count": 3              # Number of objects detected
}
```

## Visual Feedback

When running with visual output, the processor draws:

- **Green Bounding Boxes**: Around detected objects
- **Green Circles**: At object centers
- **Yellow Circle**: At hand position
- **Magenta Line**: Connecting hand to closest object
- **Distance Label**: Showing pixel distance to object

## Distance Thresholds

The processor uses the following distance descriptions based on pixel distance relative to frame diagonal:

- **Very Close**: < 10% of frame diagonal
- **Close**: 10-20% of frame diagonal
- **Medium Distance**: 20-35% of frame diagonal
- **Far Away**: > 35% of frame diagonal

## Directional Guidance

The processor provides directional guidance based on the offset between hand and object:

- **Horizontal**: "left" or "right" if offset > 30 pixels
- **Vertical**: "up" or "down" if offset > 30 pixels
- **Combined**: "left and down", "right and up", etc.

## Spatial Audio

The `pan` value in the response indicates the stereo position:

- **-1.0**: Far left
- **-0.5**: Left
- **0.0**: Center
- **0.5**: Right
- **1.0**: Far right

This can be used by the audio system to pan the TTS output, helping users locate objects by sound direction.

## Integration with Web Client

The web client (`client/screen_wss.html`) can use the `pan` value to provide spatial audio feedback:

```javascript
// Example usage in web client
if (result.pan !== undefined) {
    playAudioWithPanning(audioData, result.pan);
}
```

## Processor Configuration

Add to `processor_config.json`:

```json
{
  "15": {
    "host": "127.0.0.1",
    "port": 8016,
    "name": "object_finder_processor",
    "conda_env": "whatsai",
    "dependencies": [],
    "expects_input": "image",
    "description": "Helps blind users find and reach objects using their hands.",
    "enabled": true
  }
}
```

## Dependencies

The processor uses these building blocks:

1. **HandTrackingProcessor** (ID: 14)
   - Tracks hand position and landmarks
   - Provides normalized coordinates
   
2. **YOLOProcessor** (Scene Object Processor, ID: 4)
   - Detects objects with bounding boxes
   - Identifies object classes

Both dependencies are CPU-optimized and require no GPU.

## Use Cases

### Object Retrieval
Help blind users find and pick up everyday objects like cups, phones, keys, etc.

### Table Navigation
Guide users to locate items on a table or desk.

### Kitchen Assistance
Help users find cooking utensils, ingredients, or containers.

### Office Work
Assist in locating office supplies, documents, or devices.

## Example Scenarios

### Scenario 1: Finding a Cup

1. User starts the processor (ID: 15)
2. Processor: "No objects detected. Move camera to scan the area."
3. User moves camera to table with cup
4. Processor: "Show your hand to start finding objects."
5. User shows hand
6. Processor: "Move hand right. Cup is far away."
7. User moves hand right
8. Processor: "Move hand right. Cup is medium distance."
9. User continues moving hand right
10. Processor: "Almost there! Cup is very close."
11. User reaches cup
12. Processor: "Object reached! Cup is right there."

### Scenario 2: Multiple Objects

When multiple objects are detected, the processor guides to the closest one:

1. Processor: "Move hand left. Phone is close."
2. (Once phone is reached)
3. Processor: "Object reached! Phone is right there."
4. (If hand moves away)
5. Processor: "Move hand up. Book is medium distance."

## Performance

- **Latency**: ~50-100ms per frame on modern CPU
- **Frame Rate**: 10-20 fps (adequate for guidance)
- **CPU Usage**: Moderate (MediaPipe + YOLO on CPU)
- **Memory**: ~500MB

## Limitations

1. **Single Hand**: Only tracks one hand at a time for simplicity
2. **Closest Object**: Always guides to the closest object (no object selection yet)
3. **2D Only**: Does not account for depth (object might be behind glass, etc.)
4. **CPU Performance**: Slower than GPU-based solutions but adequate for guidance

## Future Enhancements

- Support for object selection (user chooses which object to find)
- Depth integration for 3D guidance
- Gesture-based object selection
- Multi-hand support for bimanual tasks
- Voice commands for object specification
- History of reached objects

## Building Block Design

This processor demonstrates how to combine multiple building blocks:

```python
class ObjectFinderProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        # Initialize building blocks
        self.hand_tracker = HandTrackingProcessor()
        self.object_detector = YOLOProcessor()
    
    def process_frame(self, frame):
        # Use building blocks
        hand_data = self.hand_tracker.get_hand_tracking_data(frame)
        objects = self._get_object_detections(frame)
        
        # Combine data to provide guidance
        guidance = self._generate_guidance(hand_data, objects)
        return frame, guidance
```

## Troubleshooting

### Issue: "Show your hand to start finding objects"

**Solution**: Ensure your hand is visible in the camera frame. The hand should be well-lit and clearly visible.

### Issue: "No objects detected"

**Solution**: 
- Point camera at objects on a contrasting background
- Ensure good lighting
- Move camera to scan the area
- Objects should be recognizable by YOLO11 (common objects work best)

### Issue: Guidance seems inaccurate

**Solution**:
- Ensure camera is stable and not shaking
- Improve lighting conditions
- Keep hand and objects in the same frame
- Adjust `guidance_threshold` parameter for your use case

### Issue: Processor responds slowly

**Solution**:
- This is expected on CPU-only systems
- Consider reducing frame rate if needed
- Ensure other processes aren't using excessive CPU

## Related Processors

- **Hand Tracking Processor (ID: 14)**: Building block for hand detection
- **Scene Object Processor (ID: 4)**: Building block for object detection
- **Camera Aiming Processor (ID: 13)**: Similar guidance for camera positioning

## References

- MediaPipe Hands: https://google.github.io/mediapipe/solutions/hands
- YOLO11: https://github.com/ultralytics/ultralytics
- Base Processor: `processors/base_processor.py`
