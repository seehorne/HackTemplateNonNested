# Object Finder Processor - Architecture

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Object Finder Processor                       │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                     Input: Video Frame                     │ │
│  └─────────────────────┬──────────────────────────────────────┘ │
│                        │                                         │
│          ┌─────────────┴──────────────┐                         │
│          │                            │                         │
│          ▼                            ▼                         │
│  ┌───────────────┐            ┌──────────────┐                 │
│  │ Hand Tracking │            │   Object     │                 │
│  │   Processor   │            │  Detection   │                 │
│  │  (MediaPipe)  │            │   (YOLO11)   │                 │
│  └───────┬───────┘            └──────┬───────┘                 │
│          │                           │                         │
│          │ Hand Position             │ Object Positions         │
│          │ (x, y, pixel)             │ (bbox, class, center)   │
│          │                           │                         │
│          └──────────┬────────────────┘                         │
│                     │                                           │
│                     ▼                                           │
│          ┌──────────────────────┐                              │
│          │  Guidance Generator  │                              │
│          │  - Calculate distance│                              │
│          │  - Determine direction│                             │
│          │  - Generate message  │                              │
│          │  - Calculate pan     │                              │
│          └──────────┬───────────┘                              │
│                     │                                           │
│                     ▼                                           │
│          ┌──────────────────────┐                              │
│          │   Visual Feedback    │                              │
│          │  - Draw bounding box │                              │
│          │  - Draw hand marker  │                              │
│          │  - Draw guide line   │                              │
│          └──────────┬───────────┘                              │
│                     │                                           │
│  ┌──────────────────┴─────────────────────────────────────┐   │
│  │              Output: Annotated Frame + Result          │   │
│  │  {                                                      │   │
│  │    "message": "Move hand right. Cup is close.",        │   │
│  │    "pan": 0.5,                                          │   │
│  │    "hand_detected": true,                               │   │
│  │    "objects_detected": true                             │   │
│  │  }                                                      │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Input Processing

```
Camera Frame (640x480 BGR)
         ↓
    Split to two paths:
         ↓
    ┌────┴────┐
    ▼         ▼
  Hand      Object
  Track     Detect
```

### 2. Hand Tracking Path

```
Video Frame
    ↓
Convert BGR → RGB
    ↓
MediaPipe Hands.process()
    ↓
Hand Landmarks (21 points)
    ↓
Calculate Hand Center
    ↓
Hand Position: {
  "center": {"x": 0.5, "y": 0.5, "pixel_x": 320, "pixel_y": 240},
  "location": "center",
  "distance": "close"
}
```

### 3. Object Detection Path

```
Video Frame
    ↓
YOLO11 Model
    ↓
Detections: [
  {
    "bbox": [x1, y1, x2, y2],
    "class_name": "cup",
    "confidence": 0.95,
    "center": (x, y)
  },
  ...
]
```

### 4. Guidance Generation

```
Hand Position + Object Detections
         ↓
Find Closest Object
         ↓
Calculate: distance = √((hx-ox)² + (hy-oy)²)
         ↓
Calculate: dx = ox - hx, dy = oy - hy
         ↓
Generate Directional Guidance:
  - if dx > 30: "right"
  - if dx < -30: "left"
  - if dy > 30: "down"
  - if dy < -30: "up"
         ↓
Calculate Distance Description:
  - normalized = distance / frame_diagonal
  - if < 0.1: "very close"
  - if < 0.2: "close"
  - if < 0.35: "medium distance"
  - else: "far away"
         ↓
Calculate Spatial Audio Pan:
  - pan = dx / (frame_width / 2)
  - clamp to [-1.0, 1.0]
         ↓
Build Message:
  "Move hand {direction}. {object} is {distance}."
```

### 5. Output Generation

```
Guidance Data
    ↓
Draw Visual Feedback
    ↓
Return: (annotated_frame, result_dict)
```

## Component Interactions

### HandTrackingProcessor Integration

```python
# Object Finder uses HandTrackingProcessor as a building block
hand_data = self.hand_tracker.get_hand_tracking_data(frame)

# Returns:
{
  "status": "hands_detected",
  "hand_count": 1,
  "hands": [
    {
      "center": {"pixel_x": 320, "pixel_y": 240},
      "location": "center",
      "distance": "close",
      ...
    }
  ]
}
```

### YOLOProcessor Integration

```python
# Object Finder uses YOLOProcessor's model directly
results = self.object_detector.model(frame)

# Extract detections with bounding boxes
for box in results[0].boxes:
  bbox = box.xyxy[0]  # [x1, y1, x2, y2]
  center = ((x1+x2)/2, (y1+y2)/2)
  class_name = results[0].names[box.cls]
```

## Algorithm Details

### Distance Calculation

```
Given:
  - Hand position: (hx, hy)
  - Object center: (ox, oy)

Calculate Euclidean distance:
  distance = √((hx - ox)² + (hy - oy)²)

Normalize by frame diagonal:
  diagonal = √(width² + height²)
  normalized = distance / diagonal

Map to description:
  if normalized < 0.10: "very close"
  if normalized < 0.20: "close"
  if normalized < 0.35: "medium distance"
  else: "far away"
```

### Direction Calculation

```
Given:
  - Offset: dx = ox - hx, dy = oy - hy
  - Threshold: 30 pixels

Horizontal direction:
  if dx > 30: "right"
  if dx < -30: "left"
  else: no horizontal guidance

Vertical direction:
  if dy > 30: "down"
  if dy < -30: "up"
  else: no vertical guidance

Combined:
  "{vertical} and {horizontal}"
  or "{vertical}"
  or "{horizontal}"
```

### Spatial Audio Calculation

```
Given:
  - dx: horizontal offset (ox - hx)
  - frame_width: width of frame

Calculate pan:
  pan = dx / (frame_width / 2)
  pan = clamp(pan, -1.0, 1.0)

Pan value meaning:
  -1.0: Far left
  -0.5: Left
   0.0: Center
   0.5: Right
   1.0: Far right
```

## State Management

### Processor State

```python
class ObjectFinderProcessor:
    def __init__(self):
        # Building block instances
        self.hand_tracker = HandTrackingProcessor()
        self.object_detector = YOLOProcessor()
        
        # Configuration
        self.target_object_class = None
        self.guidance_threshold = 50
        
        # State tracking
        self.last_guidance_message = ""
        self.target_object = None
```

### State Transitions

```
State: No Hand, No Objects
  ↓ (hand detected)
State: Hand Detected, No Objects
  ↓ (objects detected)
State: Hand Detected, Objects Detected
  ↓ (guidance generated)
State: Guiding (hand → object)
  ↓ (distance < threshold)
State: Reached Object
  ↓ (hand moves away)
State: Guiding (to new object)
```

## Performance Characteristics

### CPU Usage

```
Component          | CPU % | Time (ms)
-------------------|-------|----------
Hand Tracking      | 30%   | 20-30
Object Detection   | 50%   | 40-60
Guidance Calc      | 5%    | 2-5
Visual Feedback    | 10%   | 5-10
Other              | 5%    | 3-5
-------------------|-------|----------
Total              | 100%  | 70-110
```

### Memory Usage

```
Component          | Memory
-------------------|--------
MediaPipe Model    | 50 MB
YOLO11 Model       | 200 MB
Frame Buffers      | 5 MB
Python Overhead    | 50 MB
Other              | 20 MB
-------------------|--------
Total              | ~325 MB
```

## Error Handling

### Error Flow

```
Input Frame
    ↓
Try: Process Frame
    ↓
Catch: Exception
    ↓
Log Error
    ↓
Return: Safe Default
    ↓
Continue Processing
```

### Fallback Behavior

```
If hand tracking fails:
  → Return: "Show your hand to start finding objects."

If object detection fails:
  → Return: "No objects detected. Move camera to scan."

If guidance calculation fails:
  → Return: Last known guidance
  → Pan: 0.0 (center)

If visual feedback fails:
  → Return: Original frame (no annotations)
```

## Extension Points

### Custom Object Classes

```python
# Filter for specific object types
processor = ObjectFinderProcessor(
    target_object_class="cup"
)
```

### Custom Thresholds

```python
# Adjust reach sensitivity
processor = ObjectFinderProcessor(
    guidance_threshold=30  # More strict
)
```

### Custom Guidance Logic

```python
class CustomObjectFinder(ObjectFinderProcessor):
    def _generate_guidance(self, hand_data, objects, w, h):
        # Override with custom logic
        pass
```

## Building Block Pattern

This processor demonstrates the building block design:

```python
# Building Block A: Hand Tracking
hand_tracker = HandTrackingProcessor()
hand_data = hand_tracker.get_hand_tracking_data(frame)

# Building Block B: Object Detection
object_detector = YOLOProcessor()
objects = object_detector.model(frame)

# Combine Building Blocks
class ObjectFinderProcessor(BaseProcessor):
    def __init__(self):
        self.hand_tracker = HandTrackingProcessor()
        self.object_detector = YOLOProcessor()
    
    def process_frame(self, frame):
        # Use both building blocks
        hand_data = self.hand_tracker.get_hand_tracking_data(frame)
        objects = self._get_object_detections(frame)
        
        # Add new functionality
        guidance = self._combine_data(hand_data, objects)
        return frame, guidance
```

## Testing Strategy

### Unit Testing

```
Test: Hand tracking with no hand
  Input: Empty frame
  Expected: "Show your hand..."

Test: Object detection with no objects
  Input: Blank background
  Expected: "No objects detected..."

Test: Distance calculation
  Input: Hand at (100, 100), Object at (200, 200)
  Expected: distance ≈ 141 pixels

Test: Direction calculation
  Input: Hand left of object
  Expected: "Move hand right"
```

### Integration Testing

```
Test: Full pipeline
  Input: Frame with hand and object
  Expected: Valid guidance message

Test: State transitions
  Input: Sequence of frames
  Expected: Smooth state changes

Test: Error handling
  Input: Invalid frame
  Expected: Graceful degradation
```

## Deployment

### Docker Container

```
Container: whatsai
  ├── Conda Env: whatsai
  │   ├── MediaPipe
  │   ├── YOLO11
  │   ├── OpenCV
  │   └── FastAPI
  │
  ├── Processor: object_finder_processor.py
  ├── Port: 8016
  └── Config: processor_config.json
```

### Service Configuration

```json
{
  "15": {
    "host": "127.0.0.1",
    "port": 8016,
    "name": "object_finder_processor",
    "conda_env": "whatsai",
    "dependencies": [],
    "expects_input": "image",
    "enabled": true
  }
}
```

## Summary

The Object Finder Processor uses a modular architecture that:

1. Leverages existing building blocks (HandTrackingProcessor, YOLOProcessor)
2. Combines their outputs intelligently
3. Provides enhanced functionality (guidance + spatial audio)
4. Maintains clean separation of concerns
5. Follows established patterns
6. Enables easy testing and extension

This architecture demonstrates how to build complex functionality by composing simple, well-defined building blocks rather than creating monolithic solutions.
