# Hand Tracking Processor - Quick Reference

## Basic Info
- **Processor ID**: 14
- **Port**: 8015
- **Purpose**: Detect and track hands for blind/low vision users
- **Dependencies**: MediaPipe Hands (CPU-only)

## Quick Start

### Standalone Use
1. Select processor ID 14 in web client
2. Show your hand in camera view
3. Listen to audio feedback about hand location
4. Adjust hand position based on guidance
5. Use for hand-based navigation or gestures

## Return Values

### Status Types
- `hands_detected` - One or more hands found ✅
- `no_hands` - No hands detected 👋

### Example Response

**Standalone use returns string:**
```python
output_frame, message = processor.process_frame(frame)
# message: "Right hand detected at center, distance close."
```

**Building block use returns full dict:**
```python
hand_data = processor.get_hand_tracking_data(frame)
# Returns:
{
    "status": "hands_detected",
    "message": "Right hand detected at center, distance close.",
    "hand_count": 1,
    "hands": [
        {
            "hand_index": 0,
            "handedness": "Right",
            "center": {"x": 0.5, "y": 0.5, "pixel_x": 320, "pixel_y": 240},
            "offset": {"x": 0.0, "y": 0.0},
            "location": "center",
            "distance": "close",
            "size_ratio": 0.15,
            "landmarks": <MediaPipe_Landmarks_Object>
        }
    ]
}
```

## Building Block Usage

### Import and Initialize
```python
from processors.hand_tracking_processor import HandTrackingProcessor

# Basic initialization
hand_tracker = HandTrackingProcessor()

# Custom confidence thresholds
hand_tracker = HandTrackingProcessor(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=2
)

# Single hand tracking (better performance)
hand_tracker = HandTrackingProcessor(max_num_hands=1)
```

### Get Hand Data
```python
# Get full hand tracking data
hand_data = hand_tracker.get_hand_tracking_data(frame)

# Check if hands detected
if hand_data['hand_count'] > 0:
    hand = hand_data['hands'][0]
    print(f"Hand at {hand['location']}, distance {hand['distance']}")
```

### Process with Visual Output
```python
# Get both visual output and message
output_frame, message = hand_tracker.process_frame(frame)
```

## Location Zones

The processor divides the frame into regions:

- **center** - Within 30% of frame center
- **left** / **right** - Horizontal positions
- **top** / **bottom** - Vertical positions
- **top-left**, **top-right** - Corner combinations
- **bottom-left**, **bottom-right** - Corner combinations

## Distance Estimation

Based on hand size relative to frame:

- **very close** - Hand fills >25% of frame
- **close** - Hand fills 12-25% of frame
- **medium** - Hand fills 5-12% of frame
- **far** - Hand fills <5% of frame

## Configuration

### Constructor Parameters
```python
HandTrackingProcessor(
    min_detection_confidence=0.7,  # 0.0-1.0, higher = fewer false positives
    min_tracking_confidence=0.7,   # 0.0-1.0, higher = more stable tracking
    max_num_hands=2                # 1 or 2 hands
)
```

### Accessing Hand Landmarks
```python
hand_data = tracker.get_hand_tracking_data(frame)
if hand_data['hand_count'] > 0:
    landmarks = hand_data['hands'][0]['landmarks']
    # landmarks contains 21 hand keypoints
    # Use for gesture recognition, finger counting, etc.
```

## Integration Patterns

### Pattern 1: Hand Centering Guide
```python
class MyProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.hand_tracker = HandTrackingProcessor(max_num_hands=1)
    
    def process_frame(self, frame):
        hand_data = self.hand_tracker.get_hand_tracking_data(frame)
        
        if hand_data['hand_count'] == 0:
            return frame, "Show your hand"
        
        hand = hand_data['hands'][0]
        if hand['location'] == 'center':
            return frame, "Hand centered! Ready to proceed."
        else:
            return frame, f"Move hand to center. Currently at {hand['location']}"
```

### Pattern 2: Combined with Camera Aiming
```python
from processors.camera_aiming_processor import CameraAimingProcessor
from processors.hand_tracking_processor import HandTrackingProcessor

class HandGuidedProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.hand_tracker = HandTrackingProcessor()
        self.camera_aiming = CameraAimingProcessor()
    
    def process_frame(self, frame):
        # First detect hand
        hand_data = self.hand_tracker.get_hand_tracking_data(frame)
        
        if hand_data['hand_count'] > 0:
            # Then aim camera at object near hand
            guidance = self.camera_aiming.get_aiming_guidance(frame)
            return frame, f"Hand detected. {guidance['message']}"
        
        return frame, "Show your hand to begin"
```

### Pattern 3: Gesture Recognition
```python
class GestureProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.hand_tracker = HandTrackingProcessor()
    
    def process_frame(self, frame):
        hand_data = self.hand_tracker.get_hand_tracking_data(frame)
        
        if hand_data['hand_count'] > 0:
            landmarks = hand_data['hands'][0]['landmarks']
            gesture = self.recognize_gesture(landmarks)
            return frame, f"Gesture: {gesture}"
        
        return frame, "No hand detected"
    
    def recognize_gesture(self, landmarks):
        # Use landmarks to detect gestures
        # e.g., count extended fingers, analyze angles, etc.
        pass
```

## Hand Data Structure

### Main Dictionary
```python
{
    "status": "hands_detected" | "no_hands",
    "message": str,              # Audio-friendly message
    "hand_count": int,           # Number of hands detected
    "hands": [...]               # List of hand dictionaries
}
```

### Hand Dictionary (each detected hand)
```python
{
    "hand_index": int,           # 0 or 1
    "handedness": str,           # "Left" or "Right"
    "center": {
        "x": float,              # Normalized 0-1
        "y": float,              # Normalized 0-1
        "pixel_x": int,          # Pixel coordinates
        "pixel_y": int
    },
    "offset": {
        "x": float,              # -1 to 1 from center
        "y": float
    },
    "location": str,             # Spatial zone
    "distance": str,             # Distance estimation
    "size_ratio": float,         # Hand area / frame area
    "landmarks": object          # MediaPipe landmarks (21 points)
}
```

## MediaPipe Landmarks

Each hand has 21 landmarks (0-20):
- **0**: Wrist
- **1-4**: Thumb (base to tip)
- **5-8**: Index finger
- **9-12**: Middle finger
- **13-16**: Ring finger
- **17-20**: Pinky finger

Access via:
```python
landmarks = hand['landmarks']
# landmarks.landmark[0] = wrist
# landmarks.landmark[8] = index finger tip
# etc.
```

## Tips

### For Best Results
- Good lighting improves detection
- Keep hand visible (not partially occluded)
- Moderate distance works best (medium range)
- Clean background helps accuracy

### Common Issues
- **No hands detected**: Check lighting, move hand to frame
- **Wrong handedness**: Ensure full hand is visible
- **Jittery tracking**: Increase `min_tracking_confidence`
- **Missed detections**: Lower `min_detection_confidence`

## Performance

- **CPU Usage**: Moderate (MediaPipe optimized)
- **Latency**: 20-50ms per frame
- **Frame Rate**: 15-30 fps on typical CPU
- **Max Hands**: Up to 2 (configurable)

## Use Cases

### Accessibility
- Guide users to position hands for interaction
- Provide spatial feedback for blind/low vision users
- Hand-based navigation of UI elements

### Building Blocks
- Base for gesture recognition systems
- Foundation for finger counting
- Enable hand-guided object detection

### Interactive Systems
- Touch-free panel navigation
- Hand pointing detection
- Multi-hand interaction tracking

## See Also
- Full documentation: [docs/HAND_TRACKING_PROCESSOR.md](HAND_TRACKING_PROCESSOR.md)
- Example implementation: [processors/hand_guidance_example_processor.py](../processors/hand_guidance_example_processor.py)
- Camera Aiming: [docs/camera_aiming_quick_reference.md](camera_aiming_quick_reference.md)
- Main README: [../README.md](../README.md)
