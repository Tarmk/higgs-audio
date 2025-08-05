#!/usr/bin/env python3
"""
Easy Japanese TTS with Voice Cloning - No Japanese Reading Required!
This script allows you to:
1. Use your existing voice profile (from Korean TTS setup)  
2. Type Japanese text in romanized form (like "konnichiwa")
3. Generate Japanese speech in your voice

Usage:
    python easy_japanese_tts.py                  # Interactive Japanese TTS
    python easy_japanese_tts.py --romanized "konnichiwa"
    python easy_japanese_tts.py --text "こんにちは"
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

# Voice configuration (reuse from Korean setup)
MY_VOICE_DIR = "my_voice"
REFERENCE_TEXT_FILE = os.path.join(MY_VOICE_DIR, "reference.txt")
REFERENCE_AUDIO_FILE = os.path.join(MY_VOICE_DIR, "reference.wav")

# Common Japanese phrases with romanization
JAPANESE_PHRASES = {
    # Greetings
    "konnichiwa": "こんにちは",
    "ohayou gozaimasu": "おはようございます",
    "konbanwa": "こんばんは",
    "sayonara": "さようなら",
    "mata ashita": "また明日",
    "hajimemashite": "はじめまして",
    
    # Basic expressions
    "arigatou gozaimasu": "ありがとうございます",
    "arigatou": "ありがとう",
    "sumimasen": "すみません",
    "gomen nasai": "ごめんなさい",
    "hai": "はい",
    "iie": "いいえ",
    "chotto matte": "ちょっと待って",
    
    # Questions
    "genki desu ka": "元気ですか",
    "ogenki desu ka": "お元気ですか",
    "namae wa nan desu ka": "名前は何ですか",
    "doko kara kimashita ka": "どこから来ましたか",
    "nani wo shite imasu ka": "何をしていますか",
    
    # Daily expressions
    "kyou wa ii tenki desu ne": "今日はいい天気ですね",
    "gohan wo tabemashita ka": "ご飯を食べましたか",
    "oyasumi nasai": "おやすみなさい",
    "itterasshai": "いってらっしゃい",
    "tadaima": "ただいま",
    "okaeri nasai": "おかえりなさい",
    
    # Polite expressions
    "yoroshiku onegaishimasu": "よろしくお願いします",
    "domo arigatou gozaimashita": "どうもありがとうございました",
    "shitsurei shimasu": "失礼します",
    "otsukaresama deshita": "お疲れ様でした",
    
    # Study/University related
    "watashi wa daigakusei desu": "私は大学生です",
    "jinkou chinou wo benkyou shitai": "人工知能を勉強したい",
    "nihongo wo benkyou shite imasu": "日本語を勉強しています",
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


def romanized_to_japanese(text: str) -> str:
    """Convert romanized Japanese to Japanese text where possible."""
    # First check if it's already Japanese (contains Hiragana, Katakana, or Kanji)
    if any('\u3040' <= char <= '\u309f' or  # Hiragana
           '\u30a0' <= char <= '\u30ff' or  # Katakana  
           '\u4e00' <= char <= '\u9faf'     # Kanji
           for char in text):
        return text
    
    # Convert common romanized phrases
    text_lower = text.lower().strip()
    if text_lower in JAPANESE_PHRASES:
        return JAPANESE_PHRASES[text_lower]
    
    # If not found, return original (might be Japanese already or mixed)
    return text


def create_japanese_voice_clone_message(japanese_text: str) -> ChatMLSample:
    """Create voice cloning message for Japanese TTS."""
    
    # Check if voice profile exists
    if not os.path.exists(REFERENCE_TEXT_FILE) or not os.path.exists(REFERENCE_AUDIO_FILE):
        raise FileNotFoundError(
            f"Voice profile not found! Please run Korean TTS setup first:\n"
            f"python easy_korean_tts.py --setup_voice\n"
            f"Missing files:\n"
            f"- Text: {REFERENCE_TEXT_FILE}\n" 
            f"- Audio: {REFERENCE_AUDIO_FILE}"
        )
    
    # Read reference text (should be English from Korean setup)
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
            content=japanese_text
        ),
    ]
    
    return ChatMLSample(messages=messages)


def generate_japanese_tts(text: str, output_file: str, engine: HiggsAudioServeEngine, 
                         temperature: float = 0.7, seed: int = 42):
    """Generate Japanese TTS with voice cloning."""
    
    print(f"🎵 Generating Japanese TTS for: {text}")
    
    # Create voice clone message  
    chat_sample = create_japanese_voice_clone_message(text)
    
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
        
        print(f"✅ Japanese TTS saved: {output_file}")
        print(f"   Duration: {duration:.2f} seconds")
        print(f"   Sample rate: {response.sampling_rate} Hz")
        return True
    else:
        print("❌ No audio was generated")
        return False


def show_common_phrases():
    """Show common Japanese phrases with romanization."""
    print("\n🇯🇵 Common Japanese Phrases (you can type the romanized version):")
    print("=" * 70)
    
    categories = {
        "Greetings": ["konnichiwa", "ohayou gozaimasu", "konbanwa", "hajimemashite"],
        "Basic": ["arigatou gozaimasu", "sumimasen", "hai", "iie"],
        "Questions": ["genki desu ka", "namae wa nan desu ka", "doko kara kimashita ka"],
        "Daily": ["kyou wa ii tenki desu ne", "oyasumi nasai", "tadaima"],
        "Study": ["watashi wa daigakusei desu", "jinkou chinou wo benkyou shitai"]
    }
    
    for category, phrases in categories.items():
        print(f"\n📝 {category}:")
        for phrase in phrases:
            japanese = JAPANESE_PHRASES[phrase]
            print(f"  {phrase:<30} → {japanese}")


def interactive_japanese_tts(engine: HiggsAudioServeEngine):
    """Interactive Japanese TTS session."""
    print("\n🎤 Easy Japanese TTS (Type romanized Japanese or paste Japanese text)")
    print("=" * 75)
    
    show_common_phrases()
    
    print(f"\n💡 You can:")
    print("- Type romanized Japanese (like 'konnichiwa')")
    print("- Paste Japanese text (ひらがな、カタカナ、漢字)")
    print("- Type 'phrases' to see common phrases again")
    print("- Type 'quit' to exit")
    print("=" * 75)
    
    counter = 1
    while True:
        try:
            # Get input
            user_input = input("\n📝 Japanese text (romanized or 日本語): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye! さようなら!")
                break
            
            if user_input.lower() == 'phrases':
                show_common_phrases()
                continue
                
            if not user_input:
                print("⚠️  Empty input. Try typing 'konnichiwa' or paste Japanese text.")
                continue
            
            # Convert romanized to Japanese if needed
            japanese_text = romanized_to_japanese(user_input)
            
            if japanese_text != user_input:
                print(f"📝 Converted: {user_input} → {japanese_text}")
            
            output_file = f"japanese_speech_{counter:03d}.wav"
            
            # Generate TTS
            if generate_japanese_tts(japanese_text, output_file, engine):
                counter += 1
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye! さようなら!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Easy Japanese TTS - No Japanese Reading Required!")
    parser.add_argument("--text", "-t", help="Japanese text to convert to speech")
    parser.add_argument("--romanized", "-r", help="Romanized Japanese text (like 'konnichiwa')")
    parser.add_argument("--input_file", "-i", help="Input Japanese text file")
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
                        help="Show common Japanese phrases with romanization")
    
    args = parser.parse_args()
    
    # Show phrases and exit
    if args.phrases:
        show_common_phrases()
        return
    
    # Check if voice profile exists (should be set up from Korean TTS)
    if not os.path.exists(REFERENCE_TEXT_FILE) or not os.path.exists(REFERENCE_AUDIO_FILE):
        print("❌ Voice profile not found!")
        print("Please run Korean TTS voice setup first:")
        print("python easy_korean_tts.py --setup_voice")
        print("\n💡 The same voice profile works for both Korean and Japanese TTS!")
        return
    
    # Setup device
    device = setup_device()
    print(f"🔧 Using device: {device}")
    
    # Load model
    print("📦 Loading Japanese TTS model... (this may take a moment)")
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
    
    # Get Japanese text input
    japanese_text = None
    if args.romanized:
        # Convert romanized to Japanese
        japanese_text = romanized_to_japanese(args.romanized)
        if japanese_text != args.romanized:
            print(f"📝 Converted: {args.romanized} → {japanese_text}")
    elif args.text:
        japanese_text = args.text
    elif args.input_file:
        if not os.path.exists(args.input_file):
            print(f"❌ Input file not found: {args.input_file}")
            return
        with open(args.input_file, 'r', encoding='utf-8') as f:
            japanese_text = f.read().strip()
    
    # Generate TTS or start interactive mode
    if japanese_text:
        # Single generation
        output_file = args.output or "japanese_output.wav"
        generate_japanese_tts(japanese_text, output_file, engine, args.temperature, args.seed)
    else:
        # Interactive mode
        interactive_japanese_tts(engine)


if __name__ == "__main__":
    main() 