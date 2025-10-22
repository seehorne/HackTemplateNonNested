#!/usr/bin/env python3
"""
Test script for the Speech Input Processor

This script validates that the speech input processor can:
1. Start up correctly
2. Accept audio input
3. Return transcriptions
4. Work as a building block for other processors
"""

import base64
import json
import sys
import os
import numpy as np
import requests
import time

# Configuration
PROCESSOR_URL = "http://127.0.0.1:8014"
TIMEOUT = 30

def generate_test_audio():
    """Generate a simple test audio signal (1 second of 440 Hz tone)"""
    sample_rate = 16000
    duration = 1.0
    frequency = 440.0  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32) * 0.3
    
    return audio, sample_rate

def test_health_check():
    """Test the health check endpoint"""
    print("\n=== Testing Health Check ===")
    try:
        response = requests.get(f"{PROCESSOR_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Health check passed")
            print(f"  Status: {data.get('status')}")
            print(f"  Model loaded: {data.get('model_loaded')}")
            print(f"  Processor type: {data.get('processor_type')}")
            return True
        else:
            print(f"✗ Health check failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False

def test_process_endpoint():
    """Test the standard /process endpoint"""
    print("\n=== Testing /process Endpoint ===")
    try:
        # Create a simple test image (1x1 pixel)
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        import cv2
        _, buffer = cv2.imencode('.jpg', test_image)
        image_b64 = base64.b64encode(buffer).decode('utf-8')
        
        payload = {
            "image": f"data:image/jpeg;base64,{image_b64}"
        }
        
        response = requests.post(
            f"{PROCESSOR_URL}/process",
            json=payload,
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            result = data.get('result', {})
            print(f"✓ /process endpoint works")
            print(f"  Message: {result.get('message')}")
            if 'capabilities' in result:
                print(f"  Capabilities: {', '.join(result['capabilities'])}")
            return True
        else:
            print(f"✗ /process endpoint failed with status {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ /process endpoint error: {e}")
        return False

def test_transcribe_endpoint():
    """Test the custom /transcribe endpoint"""
    print("\n=== Testing /transcribe Endpoint ===")
    try:
        # Generate test audio
        audio, sample_rate = generate_test_audio()
        
        # Convert to bytes
        audio_bytes = audio.tobytes()
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        payload = {
            "audio": audio_b64,
            "format": "raw"
        }
        
        print(f"  Sending {len(audio_bytes)} bytes of audio data...")
        response = requests.post(
            f"{PROCESSOR_URL}/transcribe",
            json=payload,
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ /transcribe endpoint works")
            print(f"  Status: {data.get('status')}")
            print(f"  Transcription: '{data.get('transcription')}'")
            return True
        else:
            print(f"✗ /transcribe endpoint failed with status {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ /transcribe endpoint error: {e}")
        return False

def wait_for_processor():
    """Wait for the processor to be ready"""
    print(f"\nWaiting for processor at {PROCESSOR_URL}...")
    for i in range(30):
        try:
            response = requests.get(f"{PROCESSOR_URL}/health", timeout=2)
            if response.status_code == 200:
                print(f"✓ Processor is ready!")
                return True
        except:
            pass
        
        if i < 29:
            print(f"  Waiting... ({i+1}/30)")
            time.sleep(2)
    
    print(f"✗ Processor did not start within timeout period")
    return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Speech Input Processor Test Suite")
    print("=" * 60)
    
    # Check if we should wait for processor to start
    if not wait_for_processor():
        print("\n⚠ Note: Make sure the processor is running before running tests")
        print("  Start it with: uvicorn processors.speech_input_processor:app --host 127.0.0.1 --port 8014")
        return 1
    
    # Run tests
    results = []
    results.append(("Health Check", test_health_check()))
    results.append(("Process Endpoint", test_process_endpoint()))
    results.append(("Transcribe Endpoint", test_transcribe_endpoint()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
