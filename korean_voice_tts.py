#!/usr/bin/env python3
"""
Korean Text-to-Speech with Voice Cloning using Higgs Audio.
This script allows you to:
1. Use your own voice as reference for cloning
2. Input Korean text (typed or from file)
3. Generate Korean speech in your voice
4. Save as high-quality WAV files

Usage:
    python korean_voice_tts.py --setup_voice    # First time setup - record your voice
    python korean_voice_tts.py                  # Interactive Korean TTS
    python korean_voice_tts.py --text "안녕하세요" --output hello.wav
"""

import os
import sys
import argparse
import base64
import soundfile as sf
import torch
from loguru import logger
from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
from boson_multimodal.data_types import Message, ChatMLSample, AudioContent

# Voice configuration
MY_VOICE_DIR = "my_voice"
REFERENCE_TEXT_FILE = os.path.join(MY_VOICE_DIR, "reference.txt")
REFERENCE_AUDIO_FILE = os.path.join(MY_VOICE_DIR, "reference.wav")

# Default Korean reference text for voice setup
DEFAULT_KOREAN_TEXT = """안녕하세요. 제 이름은 한국어 음성 합성 시스템입니다. 
이 문장을 자연스럽게 읽어주세요. 
오늘 날씨가 정말 좋네요. 음성 복제 기술이 정말 놀랍습니다."""


def setup_device():
    """Automatically detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def encode_base64_content_from_file(file_path: str) -> str:
    """Encode audio file to base64."""
    with open(file_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    return audio_base64


def setup_voice_profile():
    """Guide user through setting up their voice profile."""
    print("🎤 한국어 음성 복제 설정 (Korean Voice Cloning Setup)")
    print("=" * 60)
    
    # Create voice directory
    os.makedirs(MY_VOICE_DIR, exist_ok=True)
    
    print("\n📝 Step 1: Reference text setup")
    print("You need to record yourself reading Korean text for voice cloning.")
    print("Here's the default Korean text to read:")
    print(f"\n📖 Text to read:\n{DEFAULT_KOREAN_TEXT}")
    
    # Ask if user wants custom text
    use_custom = input("\nDo you want to use custom Korean text instead? (y/n): ").strip().lower()
    
    if use_custom == 'y':
        print("\n✏️  Enter your custom Korean text:")
        print("(Make sure it's at least 2-3 sentences for better voice cloning)")
        custom_text = input("Korean text: ").strip()
        if custom_text:
            reference_text = custom_text
        else:
            print("Using default text since no input provided.")
            reference_text = DEFAULT_KOREAN_TEXT
    else:
        reference_text = DEFAULT_KOREAN_TEXT
    
    # Save reference text
    with open(REFERENCE_TEXT_FILE, "w", encoding="utf-8") as f:
        f.write(reference_text)
    
    print(f"\n💾 Reference text saved to: {REFERENCE_TEXT_FILE}")
    
    print("\n🎙️  Step 2: Audio recording")
    print("Now you need to record yourself reading the text above.")
    print(f"Save your recording as: {REFERENCE_AUDIO_FILE}")
    print("\n📋 Recording tips:")
    print("- Use good quality microphone")
    print("- Record in quiet environment") 
    print("- Speak clearly and naturally")
    print("- Keep consistent volume")
    print("- Save as WAV format (24kHz recommended)")
    
    print(f"\n⏳ Waiting for audio file at: {REFERENCE_AUDIO_FILE}")
    print("Press Enter when you've saved your recording...")
    input()
    
    # Check if file exists
    if os.path.exists(REFERENCE_AUDIO_FILE):
        print("✅ Voice profile setup complete!")
        print("You can now use Korean TTS with your voice.")
        return True
    else:
        print(f"❌ Audio file not found at: {REFERENCE_AUDIO_FILE}")
        print("Please record and save your audio file, then run setup again.")
        return False


def create_korean_voice_clone_message(korean_text: str) -> ChatMLSample:
    """Create voice cloning message for Korean TTS."""
    
    # Check if voice profile exists
    if not os.path.exists(REFERENCE_TEXT_FILE) or not os.path.exists(REFERENCE_AUDIO_FILE):
        raise FileNotFoundError(
            f"Voice profile not found! Please run with --setup_voice first.\n"
            f"Missing files:\n"
            f"- Text: {REFERENCE_TEXT_FILE}\n" 
            f"- Audio: {REFERENCE_AUDIO_FILE}"
        )
    
    # Read reference text
    with open(REFERENCE_TEXT_FILE, "r", encoding="utf-8") as f:
        reference_text = f.read().strip()
    
    # Encode reference audio
    reference_audio = encode_base64_content_from_file(REFERENCE_AUDIO_FILE)
    
    # Create voice cloning conversation pattern
    messages = [
        Message(
            role="user",
            content=reference_text,
        ),
        Message(
            role="assistant", 
            content=AudioContent(raw_audio=reference_audio, audio_url="placeholder"),
        ),
        Message(
            role="user",
            content=korean_text
        ),
    ]
    
    return ChatMLSample(messages=messages)


def generate_korean_tts(text: str, output_file: str, engine: HiggsAudioServeEngine, 
                       temperature: float = 0.7, seed: int = 42):
    """Generate Korean TTS with voice cloning."""
    
    print(f"🎵 Generating Korean TTS for: {text}")
    
    # Create voice clone message
    chat_sample = create_korean_voice_clone_message(text)
    
    # Generate audio
    response = engine.generate(
        chat_ml_sample=chat_sample,
        max_new_tokens=2048,
        temperature=temperature,
        top_k=50,
        top_p=0.95,
        seed=seed,
        force_audio_gen=True,
    )
    
    if response.audio is not None:
        # Save audio
        sf.write(output_file, response.audio, response.sampling_rate)
        duration = len(response.audio) / response.sampling_rate
        
        print(f"✅ Korean TTS saved: {output_file}")
        print(f"   Duration: {duration:.2f} seconds")
        print(f"   Sample rate: {response.sampling_rate} Hz")
        return True
    else:
        print("❌ No audio was generated")
        return False


def interactive_korean_tts(engine: HiggsAudioServeEngine):
    """Interactive Korean TTS session."""
    print("\n🎤 한국어 음성 합성 (Korean TTS with Your Voice)")
    print("=" * 50)
    print("Type Korean text and press Enter to generate speech.")
    print("Commands: 'quit', 'exit', 'q' to stop")
    print("=" * 50)
    
    counter = 1
    while True:
        try:
            # Get Korean text input
            korean_text = input("\n📝 한국어 텍스트 입력: ").strip()
            
            if korean_text.lower() in ['quit', 'exit', 'q', '종료']:
                print("👋 안녕히 가세요! (Goodbye!)")
                break
                
            if not korean_text:
                print("⚠️  빈 텍스트입니다. 한국어를 입력해주세요.")
                continue
            
            output_file = f"korean_speech_{counter:03d}.wav"
            
            # Generate TTS
            if generate_korean_tts(korean_text, output_file, engine):
                counter += 1
                
        except KeyboardInterrupt:
            print("\n👋 안녕히 가세요! (Goodbye!)")
            break
        except Exception as e:
            print(f"❌ 오류 발생: {e}")


def main():
    parser = argparse.ArgumentParser(description="Korean TTS with Voice Cloning")
    parser.add_argument("--setup_voice", action="store_true", 
                        help="Setup your voice profile for cloning")
    parser.add_argument("--text", "-t", help="Korean text to convert to speech")
    parser.add_argument("--input_file", "-i", help="Input Korean text file")
    parser.add_argument("--output", "-o", help="Output WAV file path")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for generation (0.0 = deterministic)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible generation")
    parser.add_argument("--model", default="bosonai/higgs-audio-v2-generation-3B-base",
                        help="Model path or name")
    parser.add_argument("--audio_tokenizer", default="bosonai/higgs-audio-v2-tokenizer",
                        help="Audio tokenizer path or name")
    
    args = parser.parse_args()
    
    # Handle voice setup
    if args.setup_voice:
        if setup_voice_profile():
            print("\n🎉 Voice setup complete! Now you can use Korean TTS.")
        else:
            print("\n❌ Voice setup failed. Please try again.")
        return
    
    # Check if voice profile exists
    if not os.path.exists(REFERENCE_TEXT_FILE) or not os.path.exists(REFERENCE_AUDIO_FILE):
        print("❌ Voice profile not found!")
        print("Please run with --setup_voice first:")
        print("python korean_voice_tts.py --setup_voice")
        return
    
    # Setup device
    device = setup_device()
    print(f"🔧 Using device: {device}")
    
    # Load model
    print("📦 Loading Korean TTS model... (this may take a moment)")
    try:
        engine = HiggsAudioServeEngine(
            model_name_or_path=args.model,
            audio_tokenizer_name_or_path=args.audio_tokenizer,
            device=device,
            torch_dtype=torch.bfloat16,
        )
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Get Korean text input
    korean_text = None
    if args.text:
        korean_text = args.text
    elif args.input_file:
        if not os.path.exists(args.input_file):
            print(f"❌ Input file not found: {args.input_file}")
            return
        with open(args.input_file, 'r', encoding='utf-8') as f:
            korean_text = f.read().strip()
    
    # Generate TTS or start interactive mode
    if korean_text:
        # Single generation
        output_file = args.output or "korean_output.wav"
        generate_korean_tts(korean_text, output_file, engine, args.temperature, args.seed)
    else:
        # Interactive mode
        interactive_korean_tts(engine)


if __name__ == "__main__":
    main() 