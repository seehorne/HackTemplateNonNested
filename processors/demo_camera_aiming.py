#!/usr/bin/env python3
"""
Demo script for Camera Aiming Processor

This script demonstrates the camera aiming processor with webcam input or test images.
Can be run standalone for testing.

Usage:
    python demo_camera_aiming.py                    # Use webcam (if available)
    python demo_camera_aiming.py --demo             # Run with synthetic demo images
    python demo_camera_aiming.py --image path.jpg   # Process a single image
"""

import sys
import argparse
import cv2
import numpy as np
from camera_aiming_processor import CameraAimingProcessor

def create_demo_image(scenario_idx=0):
    """Create demo images showing different scenarios"""
    scenarios = [
        ("Centered", 0.6, 0, 0),
        ("Too Far", 0.3, 0, 0),
        ("Too Close", 0.85, 0, 0),
        ("Move Left", 0.6, 100, 0),
        ("Move Right", 0.6, -100, 0),
        ("Move Up", 0.6, 0, 80),
        ("Move Down", 0.6, 0, -80),
    ]
    
    idx = scenario_idx % len(scenarios)
    name, scale, offset_x, offset_y = scenarios[idx]
    
    # Create image
    width, height = 640, 480
    image = np.ones((height, width, 3), dtype=np.uint8) * 200
    
    # Document
    doc_w = int(width * scale)
    doc_h = int(height * scale * 1.4)
    
    x = (width - doc_w) // 2 + offset_x
    y = (height - doc_h) // 2 + offset_y
    
    # Draw document
    cv2.rectangle(image, (x, y), (x + doc_w, y + doc_h), (255, 255, 255), -1)
    cv2.rectangle(image, (x, y), (x + doc_w, y + doc_h), (0, 0, 0), 3)
    
    # Add text lines
    for i in range(5):
        line_y = y + 30 + i * 40
        if line_y < y + doc_h - 20:
            cv2.line(image, (x + 20, line_y), (x + doc_w - 20, line_y), (0, 0, 0), 2)
    
    # Add scenario label
    cv2.putText(image, f"Demo: {name}", (10, height - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return image, name

def demo_mode():
    """Run demo with synthetic images"""
    print("Camera Aiming Processor - Demo Mode")
    print("=" * 60)
    print("Press 'n' for next scenario, 'q' to quit")
    print()
    
    processor = CameraAimingProcessor()
    scenario_idx = 0
    
    while True:
        # Create demo image
        demo_image, scenario_name = create_demo_image(scenario_idx)
        
        # Process
        output_frame, result = processor.process_frame(demo_image)
        
        # Print results
        print(f"\nScenario: {scenario_name}")
        print(f"Guidance: {result.get('guidance', 'N/A')}")
        if result.get('document_detected'):
            print(f"Coverage: {result.get('coverage', 'N/A')}")
        
        # Display
        cv2.imshow('Camera Aiming Demo', output_frame)
        
        # Wait for key
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            scenario_idx += 1
        else:
            scenario_idx += 1
    
    cv2.destroyAllWindows()

def webcam_mode():
    """Run with live webcam input"""
    print("Camera Aiming Processor - Webcam Mode")
    print("=" * 60)
    print("Press 'q' to quit")
    print()
    
    processor = CameraAimingProcessor()
    
    # Try to open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        print("Please use --demo mode instead")
        return
    
    print("Webcam opened successfully")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Process frame
        output_frame, result = processor.process_frame(frame)
        
        # Display guidance on frame
        guidance = result.get('guidance', 'Processing...')
        print(f"\rGuidance: {guidance:<50}", end='', flush=True)
        
        # Show frame
        cv2.imshow('Camera Aiming - Webcam', output_frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nWebcam closed")

def image_mode(image_path):
    """Process a single image file"""
    print(f"Camera Aiming Processor - Image Mode")
    print("=" * 60)
    print(f"Processing: {image_path}")
    print()
    
    processor = CameraAimingProcessor()
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Process
    output_frame, result = processor.process_frame(image)
    
    # Print results
    print("Results:")
    print(f"  Guidance: {result.get('guidance', 'N/A')}")
    print(f"  Document Detected: {result.get('document_detected', False)}")
    if result.get('document_detected'):
        print(f"  Coverage: {result.get('coverage', 'N/A')}")
        print(f"  Center Offset X: {result.get('center_offset_x', 'N/A')}")
        print(f"  Center Offset Y: {result.get('center_offset_y', 'N/A')}")
        print(f"  Well Framed: {result.get('well_framed', False)}")
    
    # Save output
    output_path = image_path.rsplit('.', 1)[0] + '_aiming_output.jpg'
    cv2.imwrite(output_path, output_frame)
    print(f"\nOutput saved to: {output_path}")
    
    # Display
    print("\nPress any key to close...")
    cv2.imshow('Camera Aiming - Result', output_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(
        description='Camera Aiming Processor Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_camera_aiming.py                    # Webcam mode (if available)
  python demo_camera_aiming.py --demo             # Demo mode with synthetic images
  python demo_camera_aiming.py --image doc.jpg    # Process single image
        """
    )
    parser.add_argument('--demo', action='store_true', 
                        help='Run demo mode with synthetic images')
    parser.add_argument('--image', type=str,
                        help='Process a single image file')
    
    args = parser.parse_args()
    
    try:
        if args.demo:
            demo_mode()
        elif args.image:
            image_mode(args.image)
        else:
            # Try webcam, fall back to demo
            try:
                webcam_mode()
            except Exception as e:
                print(f"\nWebcam not available: {e}")
                print("Falling back to demo mode...\n")
                demo_mode()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
