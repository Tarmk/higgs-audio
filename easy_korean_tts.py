#!/usr/bin/env python3
"""
Easy Korean TTS with Voice Cloning - No Korean Reading Required!
This script allows you to:
1. Record your voice in ENGLISH (which you can read)
2. Type Korean text in romanized form (like "annyeonghaseyo")
3. Generate Korean speech in your voice

Usage:
    python easy_korean_tts.py --setup_voice    # Record your voice in English
    python easy_korean_tts.py                  # Interactive Korean TTS
    python easy_korean_tts.py --romanized "annyeonghaseyo"
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

# English reference text for voice setup (easy to read!)
ENGLISH_REFERENCE_TEXT = """Hello, my name is the Korean text-to-speech system. 
Please read this sentence naturally and clearly. 
The weather is really nice today. Voice cloning technology is amazing."""

# Common Korean phrases with romanization
KOREAN_PHRASES = {
    # Greetings
    "annyeonghaseyo": "안녕하세요",
    "annyeonghi gaseyo": "안녕히 가세요",
    "manaseo bangapseumnida": "만나서 반갑습니다",
    
    # Basic phrases  
    "gamsahamnida": "감사합니다",
    "mianhamnida": "미안합니다",
    "joesonghamnida": "죄송합니다",
    "ne": "네",
    "aniyo": "아니요",
    
    # Questions
    "eotteoke jinaeseyo": "어떻게 지내세요",
    "mueoseul hago isseoyo": "무엇을 하고 있어요",
    "eodieseo wasseoyo": "어디에서 왔어요",
    
    # Daily expressions
    "oneul nalssi joa": "오늘 날씨 좋아",
    "bap meogeosseoyo": "밥 먹었어요",
    "jal jayo": "잘 자요",
    "saranghae": "사랑해",
}


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


def romanized_to_korean(text: str) -> str:
    """Convert romanized Korean to Korean text where possible."""
    # First check if it's already Korean (contains Hangul)
    if any('\uac00' <= char <= '\ud7af' for char in text):
        return text
    
    # Convert common romanized phrases
    text_lower = text.lower().strip()
    if text_lower in KOREAN_PHRASES:
        return KOREAN_PHRASES[text_lower]
    
    # If not found, return original (might be Korean already or mixed)
    return text


def setup_voice_profile():
    """Guide user through setting up their voice profile using English."""
    print("🎤 Easy Korean TTS Voice Setup (English Recording)")
    print("=" * 55)
    
    # Create voice directory
    os.makedirs(MY_VOICE_DIR, exist_ok=True)
    
    print("\n📝 Step 1: You'll record your voice in ENGLISH")
    print("(Don't worry - you don't need to know Korean!)")
    print("\n📖 Read this English text clearly:")
    print("-" * 40)
    print(ENGLISH_REFERENCE_TEXT)
    print("-" * 40)
    
    # Ask if user wants custom English text
    use_custom = input("\nUse custom English text instead? (y/n): ").strip().lower()
    
    if use_custom == 'y':
        print("\n✏️  Enter your custom English text:")
        print("(2-3 sentences work best for voice cloning)")
        custom_text = input("English text: ").strip()
        if custom_text:
            reference_text = custom_text
        else:
            print("Using default English text.")
            reference_text = ENGLISH_REFERENCE_TEXT
    else:
        reference_text = ENGLISH_REFERENCE_TEXT
    
    # Save reference text
    with open(REFERENCE_TEXT_FILE, "w", encoding="utf-8") as f:
        f.write(reference_text)
    
    print(f"\n💾 Reference text saved to: {REFERENCE_TEXT_FILE}")
    
    print("\n🎙️  Step 2: Record your voice")
    print("Now record yourself reading the ENGLISH text above.")
    print(f"Save your recording as: {REFERENCE_AUDIO_FILE}")
    print("\n📋 Recording tips:")
    print("- Use good quality microphone")
    print("- Record in quiet environment") 
    print("- Speak clearly and naturally in English")
    print("- 10-15 seconds is perfect")
    print("- Save as WAV format")
    
    print(f"\n⏳ Waiting for audio file at: {REFERENCE_AUDIO_FILE}")
    print("Press Enter when you've saved your English recording...")
    input()
    
    # Check if file exists
    if os.path.exists(REFERENCE_AUDIO_FILE):
        print("✅ Voice profile setup complete!")
        print("Now you can generate Korean speech in your voice!")
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
    
    # Read reference text (English)
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


def show_common_phrases():
    """Show common Korean phrases with romanization."""
    print("\n🇰🇷 Common Korean Phrases (you can type the romanized version):")
    print("=" * 60)
    
    categories = {
        "Greetings": ["annyeonghaseyo", "annyeonghi gaseyo", "manaseo bangapseumnida"],
        "Basic": ["gamsahamnida", "mianhamnida", "ne", "aniyo"],
        "Questions": ["eotteoke jinaeseyo", "mueoseul hago isseoyo"],
        "Expressions": ["oneul nalssi joa", "saranghae", "jal jayo"]
    }
    
    for category, phrases in categories.items():
        print(f"\n📝 {category}:")
        for phrase in phrases:
            korean = KOREAN_PHRASES[phrase]
            print(f"  {phrase:<25} → {korean}")


def interactive_korean_tts(engine: HiggsAudioServeEngine):
    """Interactive Korean TTS session."""
    print("\n🎤 Easy Korean TTS (Type romanized Korean or paste Korean text)")
    print("=" * 65)
    
    show_common_phrases()
    
    print(f"\n💡 You can:")
    print("- Type romanized Korean (like 'annyeonghaseyo')")
    print("- Paste Korean text (한글)")
    print("- Type 'phrases' to see common phrases again")
    print("- Type 'quit' to exit")
    print("=" * 65)
    
    counter = 1
    while True:
        try:
            # Get input
            user_input = input("\n📝 Korean text (romanized or 한글): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye! 안녕히 가세요!")
                break
            
            if user_input.lower() == 'phrases':
                show_common_phrases()
                continue
                
            if not user_input:
                print("⚠️  Empty input. Try typing 'annyeonghaseyo' or paste Korean text.")
                continue
            
            # Convert romanized to Korean if needed
            korean_text = romanized_to_korean(user_input)
            
            if korean_text != user_input:
                print(f"📝 Converted: {user_input} → {korean_text}")
            
            output_file = f"korean_speech_{counter:03d}.wav"
            
            # Generate TTS
            if generate_korean_tts(korean_text, output_file, engine):
                counter += 1
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye! 안녕히 가세요!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Easy Korean TTS - No Korean Reading Required!")
    parser.add_argument("--setup_voice", action="store_true", 
                        help="Setup your voice profile (record in English)")
    parser.add_argument("--text", "-t", help="Korean text to convert to speech")
    parser.add_argument("--romanized", "-r", help="Romanized Korean text (like 'annyeonghaseyo')")
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
    parser.add_argument("--phrases", action="store_true",
                        help="Show common Korean phrases with romanization")
    
    args = parser.parse_args()
    
    # Show phrases and exit
    if args.phrases:
        show_common_phrases()
        return
    
    # Handle voice setup
    if args.setup_voice:
        if setup_voice_profile():
            print("\n🎉 Voice setup complete! Now you can use Korean TTS.")
            print("Try: python easy_korean_tts.py --romanized 'annyeonghaseyo'")
        else:
            print("\n❌ Voice setup failed. Please try again.")
        return
    
    # Check if voice profile exists
    if not os.path.exists(REFERENCE_TEXT_FILE) or not os.path.exists(REFERENCE_AUDIO_FILE):
        print("❌ Voice profile not found!")
        print("Please run voice setup first:")
        print("python easy_korean_tts.py --setup_voice")
        print("\n💡 This will record your voice in ENGLISH (no Korean needed!)")
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
    if args.romanized:
        # Convert romanized to Korean
        korean_text = romanized_to_korean(args.romanized)
        if korean_text != args.romanized:
            print(f"📝 Converted: {args.romanized} → {korean_text}")
    elif args.text:
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