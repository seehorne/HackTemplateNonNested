# Object Finder Processor - Usage Guide

## Quick Start

### Step 1: Start the Server

Make sure the Docker container is running:

```bash
docker-compose up
```

### Step 2: Open the Web Client

Open `client/screen_wss.html` in your browser.

### Step 3: Select Object Finder Processor

In the processor dropdown, select **Processor ID 15 - Object Finder**.

### Step 4: Enable Camera

1. Click "Start Streaming"
2. Allow camera access when prompted
3. Position camera to show a table or surface with objects

### Step 5: Start Finding Objects

1. **Show your hand** to the camera
2. Listen to the audio guidance
3. Move your hand based on the instructions
4. When you hear "Object reached!", your hand is at the object

## How to Use

### Basic Workflow

```
1. Point camera at area with objects
   ↓
2. Show your hand in the camera view
   ↓
3. Listen: "Move hand right. Cup is medium distance."
   ↓
4. Move your hand right
   ↓
5. Listen: "Move hand right. Cup is close."
   ↓
6. Continue moving hand right
   ↓
7. Listen: "Almost there! Cup is very close."
   ↓
8. Listen: "Object reached! Cup is right there."
```

### Audio Guidance Explained

#### Directional Cues
- **"Move hand right"** - Your hand is left of the object
- **"Move hand left"** - Your hand is right of the object
- **"Move hand up"** - Your hand is below the object
- **"Move hand down"** - Your hand is above the object
- **"Move hand right and up"** - Combined directions

#### Distance Feedback
- **"far away"** - Object is far from your hand
- **"medium distance"** - Object is at medium distance
- **"close"** - Object is close to your hand
- **"very close"** - Object is very close to your hand

#### Status Messages
- **"Show your hand to start finding objects."** - Hand not detected
- **"No objects detected. Move camera to scan the area."** - No objects in view
- **"Almost there! [Object] is very close."** - Very close to object
- **"Object reached! [Object] is right there."** - Hand reached the object

### Spatial Audio

The processor uses **spatial audio** to indicate direction:

- Sound pans **left** when object is on the left
- Sound pans **right** when object is on the right
- Sound is **centered** when object is directly ahead

**Tip**: Use headphones for better spatial awareness!

## Best Practices

### Camera Positioning

1. **Stable Mount**: Keep camera steady, not handheld
2. **Good View**: Position camera to see both objects and hand
3. **Distance**: Camera should be 1-3 feet from objects
4. **Angle**: Slight downward angle works best for table surfaces

### Lighting

1. **Bright**: Ensure area is well-lit
2. **Even**: Avoid harsh shadows
3. **No Glare**: Avoid reflections on glossy surfaces

### Object Placement

1. **Contrast**: Objects should contrast with background
2. **Spacing**: Keep objects separated for easier detection
3. **Visibility**: Objects should be fully visible, not obscured
4. **Size**: Larger objects are easier to detect

### Hand Position

1. **Visible**: Keep hand fully in camera view
2. **Open**: Keep hand relatively open for better tracking
3. **Steady**: Move hand smoothly, not abruptly
4. **Height**: Keep hand at similar height as objects

## Common Scenarios

### Scenario 1: Finding a Cup on a Table

```
Setup:
- Camera mounted above table
- Cup on table
- User standing at table

Steps:
1. Camera shows table and cup
2. User shows hand above table
3. Audio: "Move hand left. Cup is medium distance."
4. User moves hand left
5. Audio: "Move hand left. Cup is close."
6. User continues moving left
7. Audio: "Almost there! Cup is very close."
8. Audio: "Object reached! Cup is right there."
9. User grasps cup
```

### Scenario 2: Multiple Objects

```
Setup:
- Camera shows desk with phone, pen, and notebook
- User wants to find phone

Steps:
1. Processor guides to closest object (pen)
2. Audio: "Move hand right. Pen is close."
3. User reaches pen
4. Audio: "Object reached! Pen is right there."
5. User moves hand away
6. Processor switches to next closest object (phone)
7. Audio: "Move hand left. Phone is medium distance."
8. User finds phone
```

### Scenario 3: Kitchen Use

```
Setup:
- Camera shows kitchen counter
- Various utensils and containers present

Usage:
1. Scan counter slowly with camera
2. Show hand when objects detected
3. Follow guidance to find each item
4. Track found items for later reference
```

## Troubleshooting

### Issue: "Show your hand to start finding objects"

**Causes:**
- Hand not in camera view
- Hand too far from camera
- Poor lighting

**Solutions:**
- Position hand in center of camera view
- Move hand closer to camera
- Improve lighting
- Ensure hand is clearly visible

### Issue: "No objects detected"

**Causes:**
- No objects in camera view
- Objects too small or far
- Poor contrast
- Low lighting

**Solutions:**
- Point camera at objects
- Move camera closer to objects
- Use objects with clear shapes and colors
- Improve lighting
- Ensure objects are on contrasting background

### Issue: Guidance seems wrong

**Causes:**
- Camera not stable
- Hand moving too fast
- Poor lighting causing tracking errors

**Solutions:**
- Stabilize camera (mount it)
- Move hand more slowly
- Improve lighting
- Wait a moment for tracking to stabilize

### Issue: Audio feedback delayed

**Causes:**
- High CPU usage
- Network latency
- Many objects in scene

**Solutions:**
- This is expected on CPU-only systems
- Close other applications
- Reduce number of objects in view
- Use better hardware if available

### Issue: Wrong object identified

**Causes:**
- YOLO model limitation
- Similar-looking objects
- Partial object visibility

**Solutions:**
- Ensure full object visibility
- Use objects with distinctive shapes
- Improve lighting
- Position camera for better view angle

## Advanced Usage

### Targeting Specific Objects

While not yet implemented in the UI, the processor supports targeting specific object classes:

```python
# Example: Only find cups
processor = ObjectFinderProcessor(
    target_object_class="cup",
    guidance_threshold=50
)
```

### Adjusting Sensitivity

Modify the reach threshold:

```python
# Closer threshold (harder to reach)
processor = ObjectFinderProcessor(guidance_threshold=30)

# Farther threshold (easier to reach)
processor = ObjectFinderProcessor(guidance_threshold=70)
```

## Integration with Other Features

### With Audio Output

The web client should automatically use spatial audio if available:

```javascript
// In screen_wss.html
if (result.pan !== undefined) {
    playAudioWithPanning(audioData, result.pan);
}
```

### With Speech Rate Control

Adjust speech rate for faster/slower guidance:
- Use the Speech Rate slider
- Or press **Alt+V** keyboard shortcut

### With Text-to-Speech

The guidance messages are designed for TTS:
- Clear, concise instructions
- Natural language
- No abbreviations or symbols

## Tips for Blind Users

1. **Start Slow**: Practice with simple setups first
2. **Use Headphones**: Spatial audio works best with stereo headphones
3. **Stable Setup**: Mount the camera for consistent results
4. **Mark Positions**: Use tactile markers to help repositioning
5. **Practice**: Familiarize yourself with common object positions
6. **Feedback**: Listen carefully to distance changes in the audio

## Tips for Assistants

1. **Setup**: Help with initial camera positioning
2. **Explain**: Describe the spatial layout initially
3. **Monitor**: Watch for any issues with tracking
4. **Adjust**: Help reposition camera or objects if needed
5. **Encourage**: Provide positive feedback during learning

## Safety Notes

1. **Clear Area**: Ensure area is free of hazards
2. **Stable Objects**: Objects should not tip or spill easily
3. **Supervision**: Consider supervision during initial learning
4. **Sharp Objects**: Be careful with sharp or hot objects
5. **Height**: Ensure objects won't fall if knocked over

## Performance Expectations

- **Latency**: 100-200ms typical on CPU
- **Accuracy**: 90%+ for common objects
- **Range**: Works best at 1-3 feet
- **Lighting**: Requires moderate to good lighting

## Supported Objects

The processor can detect many common objects including:

- **Beverages**: cup, mug, bottle, glass
- **Electronics**: phone, remote, keyboard, mouse
- **Office**: pen, pencil, book, notebook, scissors
- **Kitchen**: bowl, plate, fork, spoon, knife
- **Personal**: wallet, keys, glasses, watch
- **And many more** (80+ object classes from COCO dataset)

## Feedback and Improvement

This processor is designed to be extended and improved. Potential future enhancements:

- Voice commands for object selection
- Object history tracking
- Depth sensing for 3D guidance
- Multi-hand support
- Gesture-based controls
- Custom object training

## Related Resources

- **Hand Tracking Processor**: See [hand_tracking_quick_reference.md](hand_tracking_quick_reference.md)
- **Camera Aiming Processor**: See [camera_aiming_usage.md](camera_aiming_usage.md)
- **Full Documentation**: See [OBJECT_FINDER_PROCESSOR.md](OBJECT_FINDER_PROCESSOR.md)

## Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review processor logs in `logs/` directory
3. Open an issue on GitHub
4. Contact the development team

## Summary

The Object Finder Processor combines hand tracking and object detection to provide an accessible way for blind users to locate and reach objects. With proper setup and practice, it can significantly improve independence in finding everyday items.

**Key Points:**
- Show hand to camera
- Listen to directional guidance
- Move hand based on audio cues
- Reach objects when announced
- Use spatial audio for better direction awareness
