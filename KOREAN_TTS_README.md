# 🇰🇷 Korean Text-to-Speech with Voice Cloning

Create Korean speech audio files using **your own voice**! This system uses the Higgs Audio model to clone your voice and generate Korean text-to-speech with your vocal characteristics.

## ✨ Features

- 🎯 **Voice Cloning**: Use your own voice for Korean TTS
- 🇰🇷 **Korean Language Support**: Natural Korean text-to-speech generation
- 🎤 **Easy Voice Setup**: Guided voice recording process
- 💻 **Interactive Mode**: Type Korean text and get instant audio
- 📁 **Batch Processing**: Convert text files to speech
- 🔧 **Hardware Optimized**: Auto-detects CUDA/Apple Silicon/CPU

## 🚀 Quick Start

### 1. Install Dependencies

First, make sure you have the required packages:

```bash
pip install sounddevice soundfile  # For audio recording (optional)
```

### 2. Set Up Your Voice Profile

**Option A: Manual Recording** (recommended)
```bash
python korean_voice_tts.py --setup_voice
```

**Option B: Use Recording Helper**
```bash
python record_voice.py  # Interactive recording tool
```

This will:
- Show you Korean text to read
- Guide you through recording your voice
- Save your voice profile for cloning

### 3. Generate Korean Speech

**Interactive Mode:**
```bash
python korean_voice_tts.py
```

**Single Generation:**
```bash
python korean_voice_tts.py --text "안녕하세요, 반갑습니다!"
```

**From Text File:**
```bash
echo "오늘 날씨가 정말 좋네요." > korean_text.txt
python korean_voice_tts.py --input_file korean_text.txt --output my_speech.wav
```

## 📋 Voice Setup Guide

### What You Need:
1. **Korean text**: Default provided, or use your own
2. **Audio recording**: 10-15 seconds of you reading the Korean text
3. **Good microphone**: Clear audio quality improves voice cloning

### Recording Tips:
- 🎙️ Use a quiet environment
- 🗣️ Speak clearly and naturally
- ⏱️ Read at normal pace (not too slow/fast)
- 🔊 Maintain consistent volume
- 📏 10-15 seconds is sufficient

### Default Korean Reference Text:
```
안녕하세요. 제 이름은 한국어 음성 합성 시스템입니다. 
이 문장을 자연스럽게 읽어주세요. 
오늘 날씨가 정말 좋네요. 음성 복제 기술이 정말 놀랍습니다.
```

## 🎮 Usage Examples

### Interactive Mode
```bash
python korean_voice_tts.py

# Then type Korean text:
📝 한국어 텍스트 입력: 안녕하세요, 오늘 하루 어떠셨나요?
🎵 Generating Korean TTS for: 안녕하세요, 오늘 하루 어떠셨나요?
✅ Korean TTS saved: korean_speech_001.wav
```

### Command Line Examples
```bash
# Simple generation
python korean_voice_tts.py --text "한국어 음성 합성 테스트입니다."

# Custom output file
python korean_voice_tts.py --text "좋은 아침입니다!" --output good_morning.wav

# Adjust generation parameters
python korean_voice_tts.py --text "안녕히 가세요!" --temperature 0.5 --seed 123

# Process text file
python korean_voice_tts.py --input_file korean_story.txt --output korean_story.wav
```

### Advanced Options
```bash
python korean_voice_tts.py \
    --text "한국어로 말해보세요" \
    --output korean_output.wav \
    --temperature 0.8 \
    --seed 42
```

## ⚙️ Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--text` | Korean text to convert | None |
| `--input_file` | Korean text file to read | None |
| `--output` | Output WAV file path | `korean_output.wav` |
| `--temperature` | Generation randomness (0.0-1.0) | 0.7 |
| `--seed` | Random seed for reproducibility | 42 |
| `--setup_voice` | Set up voice profile | False |

## 📂 File Structure

After setup, you'll have:
```
higgs-audio/
├── korean_voice_tts.py       # Main Korean TTS script
├── record_voice.py           # Audio recording helper
├── my_voice/                 # Your voice profile
│   ├── reference.txt         # Korean reference text
│   └── reference.wav         # Your voice recording
└── korean_speech_*.wav       # Generated Korean audio files
```

## 🔧 Troubleshooting

### Voice Setup Issues

**Problem**: "Voice profile not found"
```bash
# Solution: Run voice setup first
python korean_voice_tts.py --setup_voice
```

**Problem**: Audio recording fails
```bash
# Install audio dependencies
pip install sounddevice soundfile

# Or record manually and save as my_voice/reference.wav
```

### Generation Issues

**Problem**: Poor voice quality
- Re-record with better microphone
- Ensure quiet recording environment  
- Use 15+ seconds of clear speech

**Problem**: Korean text not working
- Make sure text is in Korean (Hangul)
- Try shorter sentences first
- Check text encoding is UTF-8

### Performance Issues

**Problem**: Slow generation
- The system auto-detects your hardware
- GPU (CUDA/Apple Silicon) is much faster than CPU
- First run downloads models (one-time only)

## 🎵 Audio Output

- **Format**: WAV files
- **Quality**: 24kHz sample rate
- **Compatibility**: Works with any audio player
- **File size**: ~1-2MB per 10 seconds of speech

## 🔄 Re-recording Your Voice

To update your voice profile:

1. **Delete existing profile:**
   ```bash
   rm -rf my_voice/
   ```

2. **Set up new profile:**
   ```bash
   python korean_voice_tts.py --setup_voice
   ```

3. **Or use recording helper:**
   ```bash
   python record_voice.py
   ```

## 💡 Tips for Best Results

### Voice Recording:
- 🎙️ **Good microphone**: USB microphone > laptop mic
- 🔇 **Quiet space**: Minimize background noise
- 📢 **Natural speech**: Don't over-enunciate
- ⏱️ **Adequate length**: 10-15 seconds minimum

### Text Input:
- 📝 **Korean text**: Use proper Hangul characters  
- 📏 **Sentence length**: 1-3 sentences work best
- 🔤 **Mixed content**: Korean + numbers/English is OK
- 📖 **Longer texts**: Split into smaller chunks

### Generation:
- 🎲 **Try different seeds**: Each seed gives variation
- 🌡️ **Adjust temperature**: Lower = more consistent, Higher = more varied
- 🔄 **Multiple attempts**: Generate several times, pick the best

## 🎯 Example Korean Texts to Try

```korean
# Greetings
안녕하세요! 만나서 반갑습니다.

# Daily conversation  
오늘 날씨가 정말 좋네요. 어떻게 지내시나요?

# Expressions
감사합니다. 정말 고맙습니다.

# Questions
이것은 무엇인가요? 어디에서 왔나요?

# Longer text
한국어 음성 합성 기술이 정말 발전했습니다. 
자연스러운 음성을 만들 수 있어서 놀랍습니다.
```

## 🔍 How It Works

1. **Voice Analysis**: Your reference audio is analyzed for vocal characteristics
2. **Text Processing**: Korean text is processed by the language model
3. **Voice Cloning**: Your vocal features are applied to the generated speech
4. **Audio Synthesis**: High-quality 24kHz audio is generated and saved

## 🤝 Need Help?

- Check the console output for detailed error messages
- Make sure your reference audio is clear and in Korean
- Try different Korean texts to test the system
- Ensure all dependencies are installed

---

Happy Korean TTS generation! �� 즐거운 한국어 음성 합성 되세요! 