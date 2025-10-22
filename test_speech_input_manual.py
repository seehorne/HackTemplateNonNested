#!/usr/bin/env python3
"""
Manual test script for Speech Input Processor

This script can be used to manually test the speech input processor
once it's deployed in the Docker environment with all dependencies.

Usage:
    conda run -n whatsai python test_speech_input_manual.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from processors.speech_input_processor import SpeechInputProcessor
    print("✓ Successfully imported SpeechInputProcessor")
except ImportError as e:
    print(f"✗ Failed to import SpeechInputProcessor: {e}")
    sys.exit(1)


def test_initialization():
    """Test processor initialization"""
    print("\n" + "="*60)
    print("Test 1: Processor Initialization")
    print("="*60)
    
    try:
        # Test with default settings
        proc = SpeechInputProcessor()
        print("✓ Default initialization successful")
        
        # Test with custom settings
        proc_custom = SpeechInputProcessor(language="en-us", model_size="small")
        print("✓ Custom initialization successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_loading():
    """Test model loading"""
    print("\n" + "="*60)
    print("Test 2: Model Loading")
    print("="*60)
    
    try:
        proc = SpeechInputProcessor()
        
        # Try to load model
        loaded = proc._ensure_model_loaded()
        
        if loaded:
            print("✓ Model loaded successfully")
            print(f"  Language: {proc.language}")
            print(f"  Model size: {proc.model_size}")
            return True
        else:
            print("⚠ Model not loaded (expected if model not downloaded)")
            print(f"  Error: {proc._load_error}")
            print("\n  This is expected if you haven't downloaded a Vosk model yet.")
            print("  See docs/SPEECH_INPUT_PROCESSOR.md for download instructions.")
            return True  # Not a failure, just not ready
            
    except Exception as e:
        print(f"✗ Model loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_basic_recognition():
    """Test basic speech recognition (with mock audio)"""
    print("\n" + "="*60)
    print("Test 3: Basic Recognition (Mock)")
    print("="*60)
    
    try:
        proc = SpeechInputProcessor()
        
        # Check if model is loaded
        if not proc._ensure_model_loaded():
            print("⚠ Skipping recognition test - model not loaded")
            return True
        
        # Create mock WAV audio (silence)
        import wave
        import io
        
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            # 1 second of silence
            silence = b'\x00\x00' * 16000
            wav_file.writeframes(silence)
        
        audio_data = wav_buffer.getvalue()
        
        # Try to recognize
        result = proc.recognize_speech(audio_data)
        
        print("✓ Recognition completed")
        print(f"  Text: '{result['text']}'")
        print(f"  Confidence: {result['confidence']}")
        print(f"  Success: {result['success']}")
        
        # With silence, we expect no text
        if not result['text']:
            print("✓ Correctly recognized silence (no text)")
        
        return True
        
    except Exception as e:
        print(f"✗ Recognition test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_command_handler():
    """Test voice command handler"""
    print("\n" + "="*60)
    print("Test 4: Voice Command Handler")
    print("="*60)
    
    try:
        proc = SpeechInputProcessor()
        
        # Test command handler creation
        commands = {
            "start": lambda: print("  Command: START"),
            "stop": lambda: print("  Command: STOP"),
            "reset": lambda: print("  Command: RESET"),
        }
        
        handler = proc.create_voice_command_handler(commands)
        print("✓ Command handler created")
        
        # The handler would need actual audio to test execution
        print("  Handler is ready to process audio")
        
        return True
        
    except Exception as e:
        print(f"✗ Command handler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_process_frame():
    """Test process_frame method"""
    print("\n" + "="*60)
    print("Test 5: Process Frame Method")
    print("="*60)
    
    try:
        import numpy as np
        
        proc = SpeechInputProcessor()
        
        # Create dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Process frame
        output_frame, result = proc.process_frame(frame)
        
        print("✓ process_frame executed")
        print(f"  Status: {result.get('status', 'unknown')}")
        print(f"  Message: {result.get('message', 'N/A')}")
        
        if result.get('status') == 'ready':
            print("✓ Processor is ready")
        elif result.get('status') == 'not_ready':
            print("⚠ Processor not ready (model not loaded)")
        
        return True
        
    except Exception as e:
        print(f"✗ Process frame test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_audio_format_handling():
    """Test different audio format handling"""
    print("\n" + "="*60)
    print("Test 6: Audio Format Handling")
    print("="*60)
    
    try:
        proc = SpeechInputProcessor()
        
        # Test 1: numpy array
        import numpy as np
        audio_array = np.zeros(16000, dtype=np.int16)
        pcm1 = proc._prepare_audio_data(audio_array)
        print("✓ Handled numpy array input")
        
        # Test 2: raw bytes
        audio_bytes = b'\x00\x00' * 8000
        pcm2 = proc._prepare_audio_data(audio_bytes)
        print("✓ Handled raw bytes input")
        
        # Test 3: WAV bytes
        import wave
        import io
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b'\x00\x00' * 8000)
        wav_bytes = wav_buffer.getvalue()
        pcm3 = proc._prepare_audio_data(wav_bytes)
        print("✓ Handled WAV format input")
        
        return True
        
    except Exception as e:
        print(f"✗ Audio format test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("Speech Input Processor Manual Test Suite")
    print("="*60)
    
    tests = [
        ("Initialization", test_initialization),
        ("Model Loading", test_model_loading),
        ("Basic Recognition", test_basic_recognition),
        ("Command Handler", test_command_handler),
        ("Process Frame", test_process_frame),
        ("Audio Format Handling", test_audio_format_handling),
    ]
    
    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All tests passed!")
        print("\nNote: Some features require a Vosk model to be downloaded.")
        print("See docs/SPEECH_INPUT_PROCESSOR.md for download instructions.")
        return 0
    else:
        print("\n⚠️  Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
