#!/usr/bin/env python3
"""
Simple audio recording script for Korean TTS voice cloning setup.
This script helps you record your voice reading Korean text.
"""

import os
import sys
import sounddevice as sd
import soundfile as sf
import numpy as np
from loguru import logger

# Configuration
SAMPLE_RATE = 24000  # 24kHz for high quality
CHANNELS = 1  # Mono
DTYPE = np.int16

MY_VOICE_DIR = "my_voice"
REFERENCE_AUDIO_FILE = os.path.join(MY_VOICE_DIR, "reference.wav")
REFERENCE_TEXT_FILE = os.path.join(MY_VOICE_DIR, "reference.txt")

# Default Korean text
DEFAULT_KOREAN_TEXT = """안녕하세요. 제 이름은 한국어 음성 합성 시스템입니다. 
이 문장을 자연스럽게 읽어주세요. 
오늘 날씨가 정말 좋네요. 음성 복제 기술이 정말 놀랍습니다."""


def list_audio_devices():
    """List available audio input devices."""
    print("🎤 Available audio input devices:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"  {i}: {device['name']} (channels: {device['max_input_channels']})")


def record_audio(duration: float = 10.0, device_id: int = None):
    """Record audio from microphone."""
    print(f"🎙️  Recording audio for {duration} seconds...")
    print("Speak now!")
    
    try:
        # Record audio
        audio_data = sd.rec(
            int(duration * SAMPLE_RATE), 
            samplerate=SAMPLE_RATE, 
            channels=CHANNELS,
            dtype=DTYPE,
            device=device_id
        )
        
        # Wait for recording to complete
        sd.wait()
        
        print("✅ Recording complete!")
        return audio_data
        
    except sd.PortAudioError as e:
        print(f"❌ Audio recording error: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None


def play_audio(audio_data, sample_rate):
    """Play recorded audio for preview."""
    try:
        print("🔊 Playing back recorded audio...")
        sd.play(audio_data, sample_rate)
        sd.wait()
        print("✅ Playback complete!")
    except Exception as e:
        print(f"❌ Playback error: {e}")


def main():
    print("🎤 Korean TTS Voice Recording Helper")
    print("=" * 40)
    
    # Create voice directory
    os.makedirs(MY_VOICE_DIR, exist_ok=True)
    
    # Show reference text
    print("\n📖 Text to read (Korean):")
    print("-" * 30)
    
    # Check if reference text file exists
    if os.path.exists(REFERENCE_TEXT_FILE):
        with open(REFERENCE_TEXT_FILE, "r", encoding="utf-8") as f:
            text_to_read = f.read().strip()
    else:
        text_to_read = DEFAULT_KOREAN_TEXT
        # Save default text
        with open(REFERENCE_TEXT_FILE, "w", encoding="utf-8") as f:
            f.write(text_to_read)
    
    print(text_to_read)
    print("-" * 30)
    
    # List audio devices
    print()
    list_audio_devices()
    
    # Get device selection
    try:
        device_input = input("\nSelect audio device ID (or press Enter for default): ").strip()
        device_id = int(device_input) if device_input else None
    except ValueError:
        device_id = None
        print("Using default audio device.")
    
    # Get recording duration
    try:
        duration_input = input("Recording duration in seconds (default: 15): ").strip()
        duration = float(duration_input) if duration_input else 15.0
    except ValueError:
        duration = 15.0
        print("Using default duration: 15 seconds")
    
    print(f"\n🎙️  Ready to record!")
    print("Tips:")
    print("- Read the Korean text above clearly and naturally")
    print("- Speak at normal pace")
    print("- Make sure you're in a quiet environment")
    input("Press Enter to start recording...")
    
    # Record audio
    audio_data = record_audio(duration, device_id)
    
    if audio_data is None:
        print("❌ Recording failed!")
        return
    
    # Preview recording
    preview = input("\n🔊 Play back recording for preview? (y/n): ").strip().lower()
    if preview == 'y':
        play_audio(audio_data, SAMPLE_RATE)
    
    # Ask if user wants to save
    save = input("\n💾 Save this recording? (y/n): ").strip().lower()
    if save == 'y':
        try:
            # Save audio file
            sf.write(REFERENCE_AUDIO_FILE, audio_data, SAMPLE_RATE)
            print(f"✅ Audio saved to: {REFERENCE_AUDIO_FILE}")
            print(f"   Sample rate: {SAMPLE_RATE} Hz")
            print(f"   Duration: {len(audio_data) / SAMPLE_RATE:.2f} seconds")
            print(f"   File size: {os.path.getsize(REFERENCE_AUDIO_FILE) / 1024:.1f} KB")
            
            print("\n🎉 Voice recording complete!")
            print("You can now use Korean TTS with your voice:")
            print("python korean_voice_tts.py")
            
        except Exception as e:
            print(f"❌ Failed to save audio: {e}")
    else:
        print("Recording not saved. Run the script again if you want to re-record.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Recording cancelled.")
    except Exception as e:
        print(f"❌ Error: {e}") 