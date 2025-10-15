# Hand Tracking Processor - Building Block Documentation

## Overview

The Hand Tracking Processor is a building block processor designed to detect and track hands in camera views, providing non-visual feedback for users who are blind or have low vision. It uses MediaPipe Hands (CPU-only) for efficient hand detection and tracking.

## Purpose

This processor serves as a reusable building block that:
- Detects presence/absence of hands in the camera view
- Provides spatial location information (left/right/up/down/center)
- Estimates hand distance (very close/close/medium/far)
- Identifies handedness (left hand vs right hand)
- Returns structured data that other processors can consume

## Key Features

- **CPU-Only**: Uses MediaPipe Hands for efficient CPU-based detection
- **Non-Visual Feedback**: Generates audio-friendly messages for accessibility
- **Building Block Design**: Exposes both simple messages and detailed data for other processors
- **Multi-Hand Detection**: Can detect up to 2 hands simultaneously
- **Spatial Awareness**: Provides location information relative to frame center

## Usage

### As a Standalone Processor

The processor can be accessed at port 8015 (as configured in processor_config.json):

```json
POST http://localhost:8015/process
{
  "image": "base64_encoded_image_data"
}
```

Response:
```json
{
  "result": "Right hand detected at center, distance close.",
  "image": "base64_encoded_processed_image_with_landmarks"
}
```

### As a Building Block (For Other Processors)

Other processors can import and use the hand tracking functionality:

```python
from processors.hand_tracking_processor import HandTrackingProcessor

# Create an instance
hand_tracker = HandTrackingProcessor(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=2
)

# Get full hand tracking data
hand_data = hand_tracker.get_hand_tracking_data(frame)
```

#### Hand Data Structure

The `get_hand_tracking_data()` method returns a dictionary with the following structure:

```python
{
    "status": "hands_detected",  # or "no_hands"
    "message": "Right hand detected at center, distance close.",
    "hand_count": 1,
    "hands": [
        {
            "hand_index": 0,
            "handedness": "Right",  # or "Left"
            "center": {
                "x": 0.5,          # Normalized 0-1
                "y": 0.5,          # Normalized 0-1
                "pixel_x": 320,    # Pixel coordinates
                "pixel_y": 240
            },
            "offset": {
                "x": 0.0,          # -1 to 1 from center
                "y": 0.0           # -1 to 1 from center
            },
            "location": "center",  # Spatial location string
            "distance": "close",   # Distance estimation
            "size_ratio": 0.15,    # Hand area / frame area
            "landmarks": <MediaPipe_Landmarks_Object>
        }
    ]
}
```

## Configuration Parameters

### Initialization Parameters

- **min_detection_confidence** (float, default: 0.7)
  - Minimum confidence threshold for initial hand detection
  - Range: 0.0 to 1.0
  - Higher values = fewer false positives but may miss some hands

- **min_tracking_confidence** (float, default: 0.7)
  - Minimum confidence threshold for tracking existing hands
  - Range: 0.0 to 1.0
  - Higher values = more stable tracking but may lose track more easily

- **max_num_hands** (int, default: 2)
  - Maximum number of hands to detect
  - Values: 1 or 2

### Location Zones

The processor divides the frame into spatial zones:

- **Center**: Within 30% of frame center
- **Left/Right**: Beyond 30% horizontally from center
- **Top/Bottom**: Beyond 30% vertically from center
- **Combined**: e.g., "top-left", "bottom-right"

### Distance Thresholds

Based on hand size relative to frame:

- **Very Close**: > 25% of frame
- **Close**: 12-25% of frame
- **Medium**: 5-12% of frame
- **Far**: < 5% of frame

## Example Use Cases

### 1. Hand Gesture Recognition Processor

```python
from processors.hand_tracking_processor import HandTrackingProcessor

class GestureRecognitionProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.hand_tracker = HandTrackingProcessor()
    
    def process_frame(self, frame):
        # Get hand data
        hand_data = self.hand_tracker.get_hand_tracking_data(frame)
        
        if hand_data['hand_count'] > 0:
            # Access landmark data for gesture recognition
            for hand in hand_data['hands']:
                landmarks = hand['landmarks']
                # Analyze finger positions, angles, etc.
                gesture = self.recognize_gesture(landmarks)
                return frame, f"Detected gesture: {gesture}"
        
        return frame, "No hands detected"
```

### 2. Hand-Guided Camera Aiming

```python
from processors.camera_aiming_processor import CameraAimingProcessor
from processors.hand_tracking_processor import HandTrackingProcessor

class HandGuidedAimingProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.camera_aiming = CameraAimingProcessor()
        self.hand_tracker = HandTrackingProcessor()
    
    def process_frame(self, frame):
        # First detect hands
        hand_data = self.hand_tracker.get_hand_tracking_data(frame)
        
        if hand_data['hand_count'] > 0:
            # Guide user to center hand
            hand = hand_data['hands'][0]
            message = f"Hand at {hand['location']}. "
            
            # Once hand is centered, switch to object detection
            if hand['location'] == 'center':
                object_guidance = self.camera_aiming.get_aiming_guidance(frame)
                message += object_guidance['message']
            else:
                message += "Move hand to center of frame."
            
            return frame, message
        
        return frame, "Show your hand to begin"
```

### 3. Interactive Panel Navigation

```python
class InteractivePanelProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.hand_tracker = HandTrackingProcessor(max_num_hands=1)
    
    def process_frame(self, frame):
        hand_data = self.hand_tracker.get_hand_tracking_data(frame)
        
        if hand_data['hand_count'] > 0:
            hand = hand_data['hands'][0]
            
            # Use hand position to navigate panel elements
            panel_element = self.get_element_at_position(
                hand['center']['x'], 
                hand['center']['y']
            )
            
            return frame, f"Hand pointing at: {panel_element}"
        
        return frame, "No hand detected"
```

## API Reference

### HandTrackingProcessor Class

#### Methods

##### `__init__(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)`
Initialize the processor with confidence thresholds and max hands to detect.

##### `process_frame(frame: np.ndarray) -> Tuple[Optional[np.ndarray], str]`
Standard BaseProcessor interface. Returns processed frame with landmarks drawn and simple audio message.

##### `get_hand_tracking_data(frame: np.ndarray) -> Dict`
**Building block interface** for other processors. Returns full hand tracking data dictionary.

##### `process_pointcloud(point_cloud_data: Dict) -> Tuple[Optional[Dict], Union[str, Dict]]`
Not implemented. Returns pass-through data with message.

## Integration with Camera Aiming Processor

The hand tracking processor is designed to complement the camera aiming processor:

```python
# Example: Two-stage process
# Stage 1: Track hand to guide user to correct area
hand_data = hand_tracker.get_hand_tracking_data(frame)
if hand_data['status'] == 'hands_detected':
    hand_location = hand_data['hands'][0]['location']
    
    # Stage 2: Once hand is in position, switch to object aiming
    if hand_location == 'center':
        aiming_data = camera_aiming.get_aiming_guidance(frame, target_class='hand')
```

## Visual Output

When processing frames, the processor draws:
- Hand landmark points (21 per hand)
- Connections between landmarks (hand skeleton)
- Hand center indicator (yellow circle)

This visual feedback is optional and can be used for debugging or sighted assistance.

## Performance

- **CPU Usage**: Moderate (MediaPipe is optimized for CPU)
- **Latency**: ~20-50ms per frame on typical CPU
- **Max Hands**: Up to 2 hands (configurable to 1 for better performance)
- **Frame Rate**: Can process 15-30 fps on moderate CPUs

## Accessibility Features

1. **Simple Audio Messages**: Each detection includes a clear, spoken-friendly message
2. **Spatial Feedback**: Uses common directional terms (left, right, top, bottom)
3. **Distance Cues**: Helps users adjust camera distance appropriately
4. **Handedness**: Identifies which hand (left/right) for clearer guidance

## Troubleshooting

### "No hands detected" when hand is visible
- Check lighting conditions (MediaPipe works best with good lighting)
- Ensure hand is not too close or too far (medium distance works best)
- Try adjusting `min_detection_confidence` (lower value for more sensitivity)

### Tracking is jittery or unstable
- Increase `min_tracking_confidence` for more stable tracking
- Ensure steady camera and hand movement
- Check for busy/cluttered backgrounds

### Wrong handedness detected
- MediaPipe determines handedness from the camera's perspective
- Ensure hand is fully visible (not partially occluded)
- Note: "Right hand" means the hand on the right side of the frame

## Future Enhancements

Potential additions that could be built on this processor:
- Finger counting and gesture recognition
- Hand pose classification (open, closed, pointing)
- Multi-hand interaction detection
- Hand trajectory tracking over time
- Integration with depth sensors for 3D hand position

## Related Processors

- **camera_aiming_processor**: Complements hand tracking for object centering
- **finger_count_processor**: Specialized finger counting based on hand detection
- **region_processor** & **camio_processor**: Include hand detection for interactive mapping

## License & Attribution

Built using MediaPipe Hands by Google (Apache License 2.0)
Part of the HackTemplateNonNested accessibility framework
