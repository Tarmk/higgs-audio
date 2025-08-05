#!/usr/bin/env python3
"""
Simple Text-to-Speech script using Higgs Audio.
Takes text input and generates a WAV file.

Usage:
    python text_to_speech.py "Hello, world!"
    python text_to_speech.py --input_file input.txt
    python text_to_speech.py --output output.wav "Your text here"
"""

import argparse
import os
import sys
import soundfile as sf
import torch
from loguru import logger
from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
from boson_multimodal.data_types import Message, ChatMLSample, TextContent


def setup_device():
    """Automatically detect and setup the best available device."""
    if torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using Apple Silicon GPU (MPS)")
    return device


def create_simple_message(text: str) -> ChatMLSample:
    """Create a simple ChatML sample with system and user messages."""
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
    parser = argparse.ArgumentParser(description="Simple Text-to-Speech using Higgs Audio")
    parser.add_argument("text", nargs="?", help="Text to convert to speech")
    parser.add_argument("--input_file", "-i", help="Input text file")
    parser.add_argument("--output", "-o", default="output.wav", help="Output WAV file path")
    parser.add_argument("--model", default="bosonai/higgs-audio-v2-generation-3B-base", 
                        help="Model path or name")
    parser.add_argument("--audio_tokenizer", default="bosonai/higgs-audio-v2-tokenizer",
                        help="Audio tokenizer path or name")
    parser.add_argument("--temperature", type=float, default=0.7, 
                        help="Temperature for generation (0.0 = deterministic)")
    parser.add_argument("--max_tokens", type=int, default=2048,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--top_p", type=float, default=0.95, 
                        help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling parameter")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible generation")
    
    args = parser.parse_args()
    
    # Get text input
    if args.text:
        text = args.text
    elif args.input_file:
        if not os.path.exists(args.input_file):
            logger.error(f"Input file {args.input_file} not found")
            sys.exit(1)
        with open(args.input_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
    else:
        logger.error("Please provide text either as argument or via --input_file")
        sys.exit(1)
    
    if not text:
        logger.error("Empty text input")
        sys.exit(1)
        
    logger.info(f"Input text: {text}")
    logger.info(f"Output file: {args.output}")
    
    # Setup device
    device = setup_device()
    
    # Initialize the serve engine
    logger.info("Loading model and tokenizers...")
    try:
        engine = HiggsAudioServeEngine(
            model_name_or_path=args.model,
            audio_tokenizer_name_or_path=args.audio_tokenizer,
            device=device,
            torch_dtype=torch.bfloat16,
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Create message
    chat_sample = create_simple_message(text)
    
    # Generate audio
    logger.info("Generating audio...")
    try:
        response = engine.generate(
            chat_ml_sample=chat_sample,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
            force_audio_gen=True,  # Ensure we generate audio
        )
        
        if response.audio is not None:
            # Save the audio
            sf.write(args.output, response.audio, response.sampling_rate)
            logger.info(f"Audio saved to: {args.output}")
            logger.info(f"Sample rate: {response.sampling_rate} Hz")
            logger.info(f"Duration: {len(response.audio) / response.sampling_rate:.2f} seconds")
            
            # Print usage stats
            if response.usage:
                logger.info(f"Token usage: {response.usage}")
        else:
            logger.error("No audio was generated")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Failed to generate audio: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 