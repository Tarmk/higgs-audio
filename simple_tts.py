#!/usr/bin/env python3
"""
Super simple interactive Text-to-Speech using Higgs Audio.
Just run the script and type your text!
"""

import os
import soundfile as sf
import torch
from loguru import logger
from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
from boson_multimodal.data_types import Message, ChatMLSample


def setup_device():
    """Automatically detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def create_message(text: str) -> ChatMLSample:
    """Create a simple message for TTS."""
    system_message = Message(
        role="system",
        content="Generate audio following instruction."
    )
    
    user_message = Message(
        role="user", 
        content=text
    )
    
    return ChatMLSample(messages=[system_message, user_message])


def main():
    print("🎤 Simple Higgs Audio Text-to-Speech")
    print("=" * 40)
    
    # Setup device
    device = setup_device()
    print(f"🔧 Using device: {device}")
    
    # Load model
    print("📦 Loading model... (this may take a moment)")
    try:
        engine = HiggsAudioServeEngine(
            model_name_or_path="bosonai/higgs-audio-v2-generation-3B-base",
            audio_tokenizer_name_or_path="bosonai/higgs-audio-v2-tokenizer",
            device=device,
            torch_dtype=torch.bfloat16,
        )
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    print("\n" + "=" * 40)
    print("Ready! Type your text and press Enter.")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 40)
    
    counter = 1
    while True:
        try:
            # Get text input
            text = input("\n📝 Enter text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if not text:
                print("⚠️  Empty input, please enter some text.")
                continue
            
            output_file = f"speech_{counter:03d}.wav"
            
            print(f"🎵 Generating audio...")
            
            # Create message and generate
            chat_sample = create_message(text)
            response = engine.generate(
                chat_ml_sample=chat_sample,
                max_new_tokens=2048,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                seed=42,
                force_audio_gen=True,
            )
            
            if response.audio is not None:
                # Save audio
                sf.write(output_file, response.audio, response.sampling_rate)
                duration = len(response.audio) / response.sampling_rate
                
                print(f"✅ Audio saved: {output_file}")
                print(f"   Duration: {duration:.2f} seconds")
                print(f"   Sample rate: {response.sampling_rate} Hz")
                
                counter += 1
            else:
                print("❌ No audio was generated")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main() 