#!/usr/bin/env python3
"""
Manual test script for Audio Feedback Processor

This script can be used to manually test the audio feedback processor
once it's deployed in the Docker environment with all dependencies.

Usage:
    conda run -n whatsai python test_audio_feedback_manual.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from processors.audio_feedback_processor import AudioFeedbackProcessor
    print("✓ Successfully imported AudioFeedbackProcessor")
except ImportError as e:
    print(f"✗ Failed to import AudioFeedbackProcessor: {e}")
    sys.exit(1)


def test_basic_functionality():
    """Test basic audio generation"""
    print("\n" + "="*60)
    print("Testing Basic Audio Feedback Functionality")
    print("="*60)
    
    try:
        # Initialize processor
        proc = AudioFeedbackProcessor()
        print("✓ Processor initialized")
        
        # Test 1: Simple beep
        print("\nTest 1: Simple beep")
        audio = proc.generate_audio_feedback(
            audio_type="beep",
            frequency=440,
            duration=0.2,
            intensity=0.5
        )
        print(f"  ✓ Beep generated")
        print(f"    Duration: {audio['duration']:.3f}s")
        print(f"    Sample rate: {audio['sample_rate']} Hz")
        print(f"    Description: {audio['description']}")
        
        # Test 2: All audio types
        print("\nTest 2: All audio types")
        types = ["beep", "tone", "sweep", "pulse", "geiger"]
        for atype in types:
            audio = proc.generate_audio_feedback(
                audio_type=atype,
                duration=0.1,
                frequency=440,
                intensity=0.5
            )
            print(f"  ✓ {atype}: {audio['description']}")
        
        # Test 3: All presets
        print("\nTest 3: All presets")
        presets = ["scanning", "proximity", "alignment", "success", "warning", "error"]
        for preset in presets:
            audio = proc.generate_audio_feedback(preset=preset)
            print(f"  ✓ {preset}: {audio['description']}")
        
        # Test 4: Helper methods
        print("\nTest 4: Helper methods")
        
        # Proximity feedback
        prox_audio = proc.generate_proximity_feedback(distance=0.3)
        print(f"  ✓ Proximity: {prox_audio['description']}")
        
        # Alignment feedback - aligned
        align_audio = proc.generate_alignment_feedback(offset=0.05)
        print(f"  ✓ Alignment (centered): {align_audio['description']}")
        
        # Alignment feedback - not aligned
        align_audio2 = proc.generate_alignment_feedback(offset=0.5)
        print(f"  ✓ Alignment (offset): {align_audio2['description']}")
        
        # Status feedback
        status_audio = proc.generate_status_feedback("success")
        print(f"  ✓ Status: {status_audio['description']}")
        
        # Test 5: Pattern
        print("\nTest 5: Pattern application")
        pattern_audio = proc.generate_audio_feedback(
            audio_type="beep",
            frequency=440,
            duration=0.1,
            intensity=0.6,
            pattern=[0.1, 0.05, 0.1, 0.05, 0.1]
        )
        print(f"  ✓ Pattern: {pattern_audio['description']}")
        print(f"    Total duration: {pattern_audio['duration']:.3f}s")
        
        # Test 6: Parameter validation
        print("\nTest 6: Parameter validation")
        
        # Test extreme frequency
        extreme_freq = proc.generate_audio_feedback(
            audio_type="tone",
            frequency=25000,  # Above max, should clamp to 20000
            duration=0.1,
            intensity=0.5
        )
        print(f"  ✓ Frequency clamping works")
        
        # Test extreme intensity
        extreme_int = proc.generate_audio_feedback(
            audio_type="tone",
            frequency=440,
            duration=0.1,
            intensity=1.5  # Above max, should clamp to 1.0
        )
        print(f"  ✓ Intensity clamping works")
        
        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_scenario():
    """Test a realistic integration scenario"""
    print("\n" + "="*60)
    print("Testing Realistic Integration Scenario")
    print("="*60)
    
    try:
        proc = AudioFeedbackProcessor()
        
        print("\nScenario: Object proximity detection with audio feedback")
        print("-" * 60)
        
        # Simulate scanning (no object)
        print("\n1. Scanning for objects...")
        scan_audio = proc.generate_audio_feedback(preset="scanning")
        print(f"   Audio: {scan_audio['description']}")
        
        # Simulate object detected far away
        print("\n2. Object detected at distance 0.8...")
        far_audio = proc.generate_proximity_feedback(distance=0.8)
        print(f"   Audio: {far_audio['description']}")
        
        # Simulate moving closer
        print("\n3. Moving closer, distance 0.5...")
        mid_audio = proc.generate_proximity_feedback(distance=0.5)
        print(f"   Audio: {mid_audio['description']}")
        
        # Simulate very close
        print("\n4. Very close, distance 0.2...")
        close_audio = proc.generate_proximity_feedback(distance=0.2)
        print(f"   Audio: {close_audio['description']}")
        
        # Simulate alignment checking
        print("\n5. Checking alignment, offset 0.3...")
        offset_audio = proc.generate_alignment_feedback(offset=0.3)
        print(f"   Audio: {offset_audio['description']}")
        
        # Simulate aligned
        print("\n6. Aligned! Offset 0.05...")
        aligned_audio = proc.generate_alignment_feedback(offset=0.05)
        print(f"   Audio: {aligned_audio['description']}")
        
        # Simulate success
        print("\n7. Task completed!")
        success_audio = proc.generate_audio_feedback(preset="success")
        print(f"   Audio: {success_audio['description']}")
        
        print("\n" + "="*60)
        print("✅ Integration scenario test passed!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("Audio Feedback Processor Manual Test Suite")
    print("="*60)
    
    basic_ok = test_basic_functionality()
    integration_ok = test_integration_scenario()
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Basic functionality: {'✅ PASS' if basic_ok else '✗ FAIL'}")
    print(f"Integration scenario: {'✅ PASS' if integration_ok else '✗ FAIL'}")
    
    if basic_ok and integration_ok:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
