# Text-to-Speech Scripts

I've created two simple scripts to convert text to speech using the Higgs Audio model:

## 🎤 Simple Interactive Script (`simple_tts.py`)

**The easiest way to get started!**

Just run the script and type your text:

```bash
python simple_tts.py
```

- Type your text and press Enter
- Audio files are automatically saved as `speech_001.wav`, `speech_002.wav`, etc.
- Type `quit` or `exit` to stop

## 🛠️ Advanced Command-Line Script (`text_to_speech.py`)

**For more control and batch processing:**

### Basic Usage:
```bash
# Generate speech from text
python text_to_speech.py "Hello, this is a test!"

# Read from a text file
python text_to_speech.py --input_file my_text.txt

# Custom output filename
python text_to_speech.py --output my_speech.wav "Hello world!"
```

### Advanced Options:
```bash
python text_to_speech.py \
    --temperature 0.8 \
    --top_p 0.9 \
    --max_tokens 4096 \
    --seed 123 \
    --output custom_name.wav \
    "Your text here"
```

### Parameters:
- `--temperature`: Controls randomness (0.0 = deterministic, 1.0 = more random)
- `--top_p`: Top-p sampling (0.95 is default)
- `--top_k`: Top-k sampling (50 is default)
- `--max_tokens`: Maximum tokens to generate (2048 is default)
- `--seed`: Random seed for reproducible results
- `--model`: Custom model path (uses default Higgs Audio model)
- `--audio_tokenizer`: Custom audio tokenizer path

## 📋 Requirements

Both scripts automatically:
- ✅ Detect your hardware (CUDA GPU / Apple Silicon / CPU)
- ✅ Download the Higgs Audio model and tokenizer (first run only)
- ✅ Handle all the complex setup

## 🎵 Output

- Audio files are saved as WAV format
- Sample rate: 24kHz (high quality)
- Compatible with any audio player

## 💡 Tips

1. **First run**: The model download may take a few minutes
2. **GPU recommended**: Much faster generation with CUDA or Apple Silicon
3. **Short texts**: Work best for quick generation
4. **Long texts**: The script handles them automatically by chunking

## 🎯 Examples

```bash
# Quick test
python simple_tts.py
# Then type: "Hello, welcome to Higgs Audio!"

# Generate from command line
python text_to_speech.py "The weather is nice today."

# Process a story
echo "Once upon a time, in a land far away..." > story.txt
python text_to_speech.py --input_file story.txt --output story.wav
```

Happy text-to-speech generation! 🎉 