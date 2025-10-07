# Camera Aiming Processor Usage Guide

## Overview

The Camera Aiming Processor (ID: 13) provides real-time camera guidance for document capture, similar to Microsoft Seeing AI's document mode. It helps users properly frame documents by detecting document edges and providing clear directional feedback.

## Features

- **CPU-Only Processing**: No GPU required, uses OpenCV's CPU-optimized algorithms
- **Real-time Document Detection**: Uses Canny edge detection and contour analysis
- **Directional Guidance**: Provides clear instructions like "move left", "move right", "move closer", "move back"
- **Visual Overlay**: Shows target zone, center crosshair, and detected document boundaries
- **Coverage Metrics**: Reports document coverage percentage and alignment status

## How It Works

### Detection Algorithm

1. **Preprocessing**: Converts image to grayscale and applies bilateral + Gaussian filtering to reduce noise
2. **Edge Detection**: Uses Canny edge detection to find document edges
3. **Contour Analysis**: Identifies the largest rectangular contour that matches document characteristics
4. **Metrics Calculation**: Computes document position, coverage, and alignment relative to frame center
5. **Guidance Generation**: Provides audio-friendly instructions based on document position

### Guidance Messages

The processor provides contextual guidance:

- **"No document detected"**: When no document edges are found
- **"Move closer"**: Document is too small (< 50% coverage)
- **"Move back"**: Document is too large (> 85% coverage)
- **"Move left/right/up/down"**: Document is off-center
- **"Good alignment"**: Document is reasonably framed
- **"Perfect! Document is well-framed"**: Optimal framing achieved

## Configuration

### Processor Parameters

```python
CameraAimingProcessor(
    target_coverage=0.6,      # Target document coverage (60%)
    min_coverage=0.3,         # Minimum coverage to detect (30%)
    max_coverage=0.85,        # Maximum coverage (85%)
    edge_threshold_low=50,    # Canny edge detection low threshold
    edge_threshold_high=150   # Canny edge detection high threshold
)
```

### Tuning for Different Environments

- **Well-lit environments**: Use default parameters
- **Low light**: Decrease `edge_threshold_low` to 30-40
- **High contrast**: Increase `edge_threshold_high` to 180-200
- **Small documents**: Decrease `min_coverage` to 0.2
- **Large documents**: Increase `max_coverage` to 0.9

## API Usage

### Endpoint

```
POST http://localhost:8014/process
```

### Request Format

```json
{
  "image": "base64_encoded_image_data"
}
```

### Response Format

```json
{
  "result": {
    "guidance": "Move left.",
    "document_detected": true,
    "coverage": "52.7%",
    "center_offset_x": "15.3%",
    "center_offset_y": "-8.2%",
    "well_framed": false
  },
  "image": "base64_encoded_processed_image_with_overlay"
}
```

## Integration Example

### Python Client

```python
import cv2
import base64
import requests

# Capture frame
frame = cv2.imread("document.jpg")

# Encode to base64
_, buffer = cv2.imencode('.jpg', frame)
image_b64 = base64.b64encode(buffer).decode('utf-8')

# Send to processor
response = requests.post(
    "http://localhost:8014/process",
    json={"image": image_b64}
)

# Parse response
result = response.json()
guidance = result['result']['guidance']
print(f"Guidance: {guidance}")

# Save output image with overlay
if 'image' in result:
    output_b64 = result['image'].split(',')[1]
    output_data = base64.b64decode(output_b64)
    with open("output.jpg", "wb") as f:
        f.write(output_data)
```

### JavaScript/WebSocket Client

See the main WhatsAI client for integration with the WebSocket streaming system.

## Best Practices

### For Optimal Detection

1. **Good Lighting**: Ensure document is well-lit with minimal shadows
2. **Contrast**: Document should contrast with background (white paper on dark table works best)
3. **Clear Edges**: Document should have clean, unobstructed edges
4. **Flat Surface**: Keep document as flat as possible
5. **Steady Camera**: Minimize camera shake for stable detection

### Common Issues

**Issue**: Document not detected
- **Cause**: Poor lighting, low contrast, or document partially out of frame
- **Solution**: Improve lighting, ensure document edges are visible, adjust edge detection thresholds

**Issue**: False positives (detects non-document objects)
- **Cause**: Other rectangular objects in frame
- **Solution**: Clear the area around document, or increase `min_coverage`

**Issue**: Unstable guidance (guidance changes rapidly)
- **Cause**: Borderline detection or camera shake
- **Solution**: Stabilize camera, ensure document is fully in frame

## Performance

- **Processing Time**: ~20-50ms per frame on typical CPU
- **Resolution**: Works with any resolution, auto-scales large images
- **Frame Rate**: Can process 20-50 FPS depending on CPU

## Limitations

1. **Document Types**: Works best with standard rectangular documents (A4, Letter, etc.)
2. **Backgrounds**: Performs best with contrasting backgrounds
3. **Rotation**: Can handle slight rotation (<20°), but perpendicular alignment is recommended
4. **Multiple Documents**: Detects only the largest/most prominent document
5. **Partial Occlusion**: May fail if document edges are significantly obscured

## Use Cases

- **Accessibility**: Guide visually impaired users to properly frame documents for OCR
- **Document Scanning Apps**: Provide real-time framing guidance
- **Photo ID Capture**: Help users center and frame ID cards
- **Receipt Scanning**: Guide proper receipt positioning
- **Form Capture**: Assist with capturing forms and questionnaires

## Future Enhancements

Possible improvements for future versions:
- Perspective correction for rotated documents
- Multi-document detection and selection
- Adaptive threshold tuning based on lighting conditions
- Blur detection to ensure sharp images
- QR code and barcode detection for guidance
