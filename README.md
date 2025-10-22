# WhatsAI Web Client

This project was created to be used by Blind and Low Vision users to have an AAII (Accessible Artificial Intelligence Implementation) available to them wherever they may be. Most components of this project have minimal dependency on a stable internet connection once all the components have been installed if the user wants to work solely on their workstation (PC/laptop). If the user wants to access it from anywhere, a WhatsApp connection needs to be set up. You would need both the server and the client web applications running for it to work. Make sure to start the server first. You can access the client by opening client/screen_wss.html in your browser.

## Quick Start with Remote Server

If you want to use an already running server:

1. Obtain the server WebSocket URL (format: `wss://[server-address]/ws`)
2. Open the WhatsAI Web Client by opening the client/screen_wss.html in your browser.
3. Paste the server URL into the "Server URL" text field
4. Click "Select Screen to Share" and choose your desired screen
5. Click "Start Streaming" to begin sharing your screen with the server
6. Select a processor from the dropdown menu to analyze your screen

## Available Processors

The system includes several processors, ordered by complexity:

### Basic Processor (ID: 0)
- **Description**: A simple pass-through processor that returns the original image without modifications
- **Dependencies**: None
- **Use Case**: Testing connectivity and stream quality

### Scene Object Processor (ID: 4)
- **Description**: Performs real-time object detection and segmentation using YOLO11
- **Dependencies**: None
- **Use Case**: Identifying and locating objects in your screen
- **Reference**: Based on Ultralytics YOLO - https://github.com/ultralytics/ultralytics

### Scene Captioning Processor (ID: 2)
- **Description**: Generates detailed region-based captions and performs OCR using Florence-2
- **Dependencies**: None
- **Use Case**: Understanding text and visual content on screen
- **Reference**: Microsoft Florence-2 - https://huggingface.co/microsoft/Florence-2-large

### CamIO Processor (ID: 3)
- **Description**: Provides specialized object recognition for curated items
- **Dependencies**: None
- **Use Case**: Learning about specific pre-trained objects
- **Reference**: Based on Simple CamIO - https://github.com/Coughlan-Lab/simple_camio

### Finger Count Processor (ID: 1)
- **Description**: Detects hands and counts raised fingers using MediaPipe
- **Dependencies**: Basic Processor (ID: 0)
- **Use Case**: Gesture recognition and finger counting
- **Reference**: Adapted from Finger Counter using MediaPipe - https://github.com/HarshitDolu/Finger-Counter-using-mediapipe

### SeeingAI Short Text Processor (ID: 12)
- **Description**: Extracts short text from images similar to Microsoft's SeeingAI, using CPU-only OCR
- **Dependencies**: None (EasyOCR automatically installed)
- **Use Case**: Quick text reading for accessibility, similar to SeeingAI's short text feature
- **Reference**: Uses EasyOCR for CPU-based text recognition
- **Features**:
  - ⭐ **Out-of-View Detection** - Stops reading text when it moves outside the viewing area (cropped region)
  - ⭐ **Adjustable Speech Rate** - Control TTS speed from 0.5x to 2.0x using Alt+V keyboard shortcut
  - ⭐ **Smart Text Repetition Prevention** - Prevents reading the same text multiple times with:
    - 10-second prevention window (won't repeat text within 10 seconds)
    - Normalized text comparison (handles minor OCR variations like extra spaces or case differences)
    - Queue and current speech checking to avoid duplicates
- **Note**: First-time startup may take longer as EasyOCR downloads models automatically

### Camera Aiming Processor (ID: 13) 🆕
- **Description**: Building block processor that helps users who are blind or low vision aim their camera to center objects using non-visual audio cues
- **Dependencies**: None (uses YOLO11 for object detection)
- **Use Case**: Taking properly framed photos or centering objects for accessibility
- **Reference**: Built as a reusable building block for other processors to incorporate camera aiming functionality
- **Features**:
  - 🎯 **Directional Guidance** - Audio cues for moving camera left/right/up/down
  - ✅ **Centering Confirmation** - Notifies when object is properly centered
  - 📏 **Distance Feedback** - Indicates if camera is too close or too far from object
  - 📐 **Size Optimization** - Ensures object fills frame appropriately (15-60% optimal for comfortable distance)
  - 🎨 **Visual Overlay** - Optional visual indicators for sighted users (crosshairs, bounding boxes)
  - 🔧 **Building Block Design** - Other processors can call `get_aiming_guidance()` to reuse functionality
- **CPU-Only**: Fully compatible with CPU-only systems, no GPU required
- **Documentation**: See [docs/camera_aiming_usage.md](docs/camera_aiming_usage.md) for detailed usage examples and integration guide

### Hand Tracking Processor (ID: 14) 🆕
- **Description**: Building block processor that detects and tracks hands in the camera view, providing spatial location feedback through non-visual audio cues
- **Dependencies**: None (uses MediaPipe Hands for CPU-only hand detection)
- **Use Case**: Hand-based navigation, gesture recognition foundation, interactive control for accessibility
- **Reference**: Built as a reusable building block for other processors requiring hand detection and tracking
- **Features**:
  - 👋 **Hand Detection** - Detects up to 2 hands simultaneously with handedness (left/right) identification
  - 📍 **Spatial Location** - Audio feedback for hand position (center, left, right, top, bottom, corners)
  - 📏 **Distance Estimation** - Relative distance feedback (very close, close, medium, far)
  - 🎯 **Positional Guidance** - Helps users center their hands in frame with directional cues
  - 🖐️ **Landmark Data** - Full 21-point hand landmark data for advanced gesture recognition
  - 🔧 **Building Block Design** - Other processors can call `get_hand_tracking_data()` to access detailed hand information
- **CPU-Only**: Uses MediaPipe Hands (fully CPU-optimized), no GPU required
- **Documentation**: See [docs/hand_tracking_quick_reference.md](docs/hand_tracking_quick_reference.md) for quick start and [docs/HAND_TRACKING_PROCESSOR.md](docs/HAND_TRACKING_PROCESSOR.md) for comprehensive guide
- **Example**: [processors/hand_guidance_example_processor.py](processors/hand_guidance_example_processor.py) demonstrates how to use as a building block

### Object Finder Processor (ID: 15) 🆕
- **Description**: Helps blind users find and reach objects with their hands using non-visual audio cues
- **Dependencies**: Uses Hand Tracking Processor (ID: 14) and Scene Object Processor (ID: 4) as building blocks
- **Use Case**: Object retrieval assistance for blind and low vision users
- **Features**:
  - 🎯 **Hand-to-Object Guidance** - Provides directional cues to guide user's hand to detected objects
  - 🏷️ **Object Identification** - Identifies which specific object the hand is near and lists nearby alternatives
  - 📏 **Distance Feedback** - Tells user how far their hand is from target object (very close, close, medium, far)
  - ✅ **Reach Confirmation** - Announces when hand reaches the target object
  - 📝 **Multiple Object Awareness** - Lists detected objects when hand not shown, mentions nearby objects during guidance
  - 🔧 **Building Block Integration** - Demonstrates how to combine multiple building block processors
- **CPU-Only**: Leverages CPU-optimized MediaPipe Hands and YOLO11 models
- **How It Works**: 
  1. Detects objects in camera view using YOLO11
  2. Lists detected objects when no hand shown: "Show your hand to find objects. I see: cup, phone, book."
  3. Tracks user's hand position using MediaPipe
  4. Calculates direction and distance from hand to closest object
  5. Provides real-time audio guidance: "Move hand right. Cup is medium distance. Phone also nearby."
  6. Announces when object is reached: "Object reached! Cup is right there. Also nearby: phone."

### Audio Feedback Processor (ID: 16) 🆕
- **Description**: Building block processor for generating non-verbal audio feedback (beeps, tones, varying pitch/frequency)
- **Dependencies**: None (pure NumPy audio generation)
- **Use Case**: Provides reusable audio feedback capabilities for other processors requiring non-verbal cues
- **Reference**: Built as a reusable building block for accessibility-focused processors
- **Features**:
  - 🔊 **Multiple Audio Types** - Beeps, tones, frequency sweeps, pulses, Geiger counter-style clicks
  - 🎵 **Configurable Parameters** - Adjustable frequency (20-20000 Hz), duration, intensity
  - 🎯 **Ready Presets** - Pre-configured audio for scanning, proximity, alignment, success, warning, error
  - 📏 **Proximity Feedback** - Distance-based audio with varying intensity and frequency
  - 📐 **Alignment Feedback** - Audio cues for centering and positioning tasks
  - 🔧 **Building Block Design** - Other processors can call `generate_audio_feedback()` and helper methods
  - 🎨 **Pattern Support** - Create complex audio patterns with pauses and repetitions
- **CPU-Only**: Pure NumPy-based audio generation, no external libraries or GPU required
- **Documentation**: See [docs/AUDIO_FEEDBACK_PROCESSOR.md](docs/AUDIO_FEEDBACK_PROCESSOR.md) for comprehensive guide and examples
- **Note**: When used directly, generates a demo beep to show it's working

### Audio Feedback Example Processor (ID: 17) 🆕
- **Description**: Example demonstrating how to use AudioFeedbackProcessor as a building block
- **Dependencies**: Uses Audio Feedback Processor (ID: 16) and YOLO for object detection
- **Use Case**: Shows integration pattern for audio feedback in custom processors
- **Features**:
  - 🎯 **Object Detection** - Detects objects using YOLO
  - 📏 **Proximity Audio** - Generates audio based on object distance
  - 📐 **Centering Audio** - Provides alignment feedback when object is centered
  - 🔧 **Integration Example** - Demonstrates building block usage pattern
- **CPU-Only**: Uses CPU-optimized YOLO and pure NumPy audio
- **Reference**: See [processors/audio_feedback_example_processor.py](processors/audio_feedback_example_processor.py) for code

### Speech Input Processor (ID: 18) 🆕
- **Description**: Building block processor for speech-to-text input using voice (speech recognition)
- **Dependencies**: None (uses Vosk for CPU-only offline speech recognition)
- **Use Case**: Provides reusable speech input capabilities for other processors requiring voice commands, dictation, or verbal navigation
- **Reference**: Built as a reusable building block for processors needing speech-based user input
- **Features**:
  - 🎤 **Offline Recognition** - CPU-only speech-to-text without internet connection (after model download)
  - 🌍 **Multiple Languages** - Support for 10+ languages including English, Spanish, French, German, etc.
  - 📊 **Confidence Scores** - Word-level and overall confidence metrics
  - ⏱️ **Word Timestamps** - Detailed timing information for each recognized word
  - 🎯 **Voice Commands** - Built-in command handler for voice-controlled applications
  - 🔧 **Building Block Design** - Other processors can call `recognize_speech()` to add voice input
  - 📝 **Multiple Modes** - Single command, continuous, and dictation modes
- **CPU-Only**: Uses Vosk offline speech recognition, fully CPU-optimized, no GPU required
- **Documentation**: See [docs/SPEECH_INPUT_PROCESSOR.md](docs/SPEECH_INPUT_PROCESSOR.md) for comprehensive guide and [docs/speech_input_quick_reference.md](docs/speech_input_quick_reference.md) for quick start
- **Note**: Requires downloading a Vosk model (~50MB for small, ~1.5GB for large) on first use. See documentation for download instructions.

### Speech Command Example Processor (ID: 19) 🆕
- **Description**: Example demonstrating how to use SpeechInputProcessor as a building block
- **Dependencies**: Uses Speech Input Processor (ID: 18) as a building block
- **Use Case**: Shows how to create voice-controlled image processing applications
- **Features**:
  - 🎤 **Voice Commands** - Control image filters using voice (grayscale, blur, edge, invert)
  - 🔧 **Integration Example** - Demonstrates speech input building block usage pattern
  - 🎯 **Command Matching** - Shows how to map voice commands to actions
- **CPU-Only**: Leverages CPU-optimized Vosk speech recognition
- **How It Works**:
  1. Accepts voice commands like "grayscale", "blur", "edge", "invert"
  2. Uses SpeechInputProcessor to recognize speech
  3. Applies corresponding image filter based on command
  4. Provides visual feedback of current filter state
- **Reference**: See [processors/speech_command_example_processor.py](processors/speech_command_example_processor.py) for code

## Setting Up Your Own Server

### Local Server Setup

#### Prerequisites
1. WSL2 on Windows or Linux system
2. Docker Desktop with WSL2 integration enabled
3. Gemini API key

#### Installation Steps

1. Clone the repository:
   ```bash
   git clone --single-branch --branch workshop https://github.com/Znasif/HackTemplate.git
   cd HackTemplate
   ```

2. Create and configure the `.env` file:
   ```bash
   GEMINI_API_KEY="your-gemini-api-key"
   ```

3. Build the Docker container:
   ```bash
   docker-compose build
   ```

4. Start the server:
   ```bash
   docker-compose up
   ```

5. Access the server:
   - Local access: `ws://localhost:8000/ws`
   - Remote access: Use localtunnel (https://github.com/localtunnel/localtunnel) to expose the port

## Troubleshooting

### Processor Connection Errors

If you see errors like "Could not connect to the processor" for any processor:

1. **Check processor logs**: Look in the `logs/` directory for processor-specific log files
2. **Verify dependencies**: Ensure all required dependencies are installed in the conda environment
3. **Check port availability**: Make sure the processor's port isn't already in use
4. **Rebuild container**: If dependencies were recently added, rebuild with `docker-compose build`

**Common fixes:**
- For SeeingAI processor: Ensure EasyOCR is available in the whatsai conda environment
- For other processors: Check the processor's specific dependencies in `resources/whatsai/pyproject.toml`

### SeeingAI Short Text Processor Issues

The SeeingAI processor (ID: 12) requires EasyOCR which downloads models on first use:

- **First startup may be slow**: EasyOCR downloads models automatically (this is normal)
- **Connection timeout**: If the processor takes too long to start, wait a few minutes and try again
- **Missing dependency**: If you see import errors, rebuild the Docker container to install EasyOCR
- **Out-of-view detection**: ⭐ Automatically detects when text moves outside the cropped viewing area
  - Use cropping controls to focus on specific screen areas
  - The processor will warn when text goes out of view
  - Helps prevent reading text that's no longer visible to the user
- **Speech rate control**: ⭐ Adjustable text-to-speech reading speed
  - Use the Speech Rate slider or Alt+V keyboard shortcut
  - Range from 0.5x (slow) to 2.0x (fast)
  - Default is 1.0x (normal speed) for comfortable listening
  - Automatically speeds up when text queue builds up

### Deployment to RunPod

#### Prerequisites
1. Docker Hub account
2. RunPod account with credits

#### Deployment Steps

1. Tag and push your local Docker image to Docker Hub:
   ```bash
   docker tag whatsai-server:latest yourusername/whatsai:latest
   docker push yourusername/whatsai:latest
   ```

2. Create a new pod on RunPod:
   - Container Image: `yourusername/whatsai:latest`
   - Container Start Command: `bash -c "cd /app && /app/start_server.sh"`
   - Container Disk: 50 GB
   - Volume Disk: 20 GB (optional, for persistent storage)
   - Volume Mount Path: `/workspace`
   - Expose HTTP Ports: `8000`
   - GPU Selection: RTX 4000 Ada or similar

3. Add environment variables in RunPod:
   - GEMINI_API_KEY: your-api-key
   - PYTHONPATH: /app

4. Deploy the pod and obtain your URL:
   - Format: `wss://[pod-id]-8000.proxy.runpod.net/ws`


## Audio Streaming Features

The system supports real-time audio streaming for voice-based interactions:

1. **Direct Audio**: Screen reader and system audio output work automatically
2. **Remote Audio via Start Audio Button**: Dictate processor by pressing the "Start Audio" button and saying which processor to start and then "Stop Audio" to initiate.
3. **Virtual Audio Cable**: For WhatsApp audio streaming:
   - Install VB-Audio Virtual Cable from https://vb-audio.com/Cable/
   - Set VB-Audio Virtual Cable Output as default system audio output
   - In WhatsApp calls, set audio input to VB-Audio Virtual Cable Input

## API Endpoints

- **WebSocket**: `ws://[server]/ws` - Main streaming endpoint
- **HTTP GET**: `http://[server]:8000/processors` - List available processors

## Troubleshooting

### Connection Issues
- Verify the server URL format includes `/ws` at the end
- Check if the server is running: `curl http://[server]:8000/processors`
- Ensure your firewall allows WebSocket connections

### Performance Optimization
- For best results, use a wired internet connection
- Close unnecessary applications to reduce screen capture overhead
- Select specific application windows instead of full screen when possible

## Brief Demo

Click on the following image which will take you to a playlist:

[![Demo Link for Whatsapp Livestream AI processing](https://i.ytimg.com/vi/ExhlwkUW_gc/hqdefault.jpg?sqp=-oaymwExCNACELwBSFryq4qpAyMIARUAAIhCGAHwAQH4Af4JgALQBYoCDAgAEAEYZSBRKEAwDw==&rs=AOn4CLDxzMwlnE3AVdbFIucWFV93J9Jg3g)](https://www.youtube.com/playlist?list=PLk3VM_Y78PILin5BQJ0cYq_OdmuT7v1VY)