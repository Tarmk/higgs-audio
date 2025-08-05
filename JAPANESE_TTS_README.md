# 🇯🇵 Easy Japanese TTS with Voice Cloning

Generate Japanese speech in your own voice **without needing to read Japanese!** 

This system uses your existing voice profile from the Korean TTS setup and allows you to type romanized Japanese (like "konnichiwa") to generate natural Japanese speech.

## ✨ Features

- 🎤 **Voice Cloning**: Uses your voice recording to generate Japanese speech
- 🔤 **Romanization Support**: Type "konnichiwa" instead of "こんにちは"  
- 🚀 **No Japanese Reading Required**: Built-in phrase dictionary
- 🎯 **High Quality**: 24kHz audio output with natural prosody
- 📚 **Common Phrases**: Pre-loaded with essential Japanese expressions

## 🚀 Quick Start

### Prerequisites
Make sure you have a voice profile set up from the Korean TTS:
```bash
# If you haven't set up voice profile yet:
python3 easy_korean_tts.py --setup_voice
```

### Basic Usage

**Simple Japanese greeting:**
```bash
python3 easy_japanese_tts.py --romanized "konnichiwa"
```

**Longer sentence:**
```bash
python3 easy_japanese_tts.py --romanized "watashi wa daigakusei desu"
```

**Direct Japanese text:**
```bash
python3 easy_japanese_tts.py --text "こんにちは、元気ですか？"
```

**Interactive mode:**
```bash
python3 easy_japanese_tts.py
```

## 📝 Common Japanese Phrases

### Greetings
| Romanized | Japanese | English |
|-----------|----------|---------|
| `konnichiwa` | こんにちは | Hello |
| `ohayou gozaimasu` | おはようございます | Good morning |
| `konbanwa` | こんばんは | Good evening |
| `hajimemashite` | はじめまして | Nice to meet you |

### Basic Expressions  
| Romanized | Japanese | English |
|-----------|----------|---------|
| `arigatou gozaimasu` | ありがとうございます | Thank you |
| `sumimasen` | すみません | Excuse me/Sorry |
| `hai` | はい | Yes |
| `iie` | いいえ | No |

### Questions
| Romanized | Japanese | English |
|-----------|----------|---------|
| `genki desu ka` | 元気ですか | How are you? |
| `namae wa nan desu ka` | 名前は何ですか | What is your name? |
| `doko kara kimashita ka` | どこから来ましたか | Where are you from? |

### Study/University
| Romanized | Japanese | English |
|-----------|----------|---------|
| `watashi wa daigakusei desu` | 私は大学生です | I am a university student |
| `jinkou chinou wo benkyou shitai` | 人工知能を勉強したい | I want to study AI |
| `nihongo wo benkyou shite imasu` | 日本語を勉強しています | I am studying Japanese |

## 🎯 Usage Examples

### Command Line Examples

**Generate introduction:**
```bash
python3 easy_japanese_tts.py --romanized "hajimemashite, watashi wa daigakusei desu" --output intro.wav
```

**Multiple sentences:**
```bash
python3 easy_japanese_tts.py --text "こんにちは。私は大学生です。人工知能を勉強したいです。" --output self_intro.wav
```

**From text file:**
```bash
echo "日本語を勉強しています。とても面白いです。" > japanese_text.txt
python3 easy_japanese_tts.py --input_file japanese_text.txt --output study.wav
```

### Interactive Mode Features

In interactive mode, you can:
- Type romanized Japanese: `konnichiwa`
- Paste Japanese text: `こんにちは`
- View phrases: `phrases`
- Mix styles: `arigatou, genki desu ka?`

## 🔧 Advanced Options

```bash
python3 easy_japanese_tts.py \
    --romanized "konnichiwa" \
    --output hello.wav \
    --temperature 0.8 \
    --seed 123
```

### Parameters:
- `--temperature`: Creativity (0.0 = deterministic, 1.0 = creative)
- `--seed`: Random seed for reproducible results
- `--model`: Use different model (default: higgs-audio-v2-generation-3B-base)

## 💡 Tips for Best Results

1. **Voice Profile**: Use clear, natural English recording for voice cloning
2. **Romanization**: Stick to standard Hepburn romanization
3. **Mixing**: You can mix romanized and Japanese text
4. **Length**: Longer sentences work well (2-3 sentences optimal)
5. **Punctuation**: Use Japanese punctuation (。、！？) for natural pauses

## 🗾 Japanese Script Support

The system automatically detects and handles:
- **Hiragana**: ひらがな (phonetic script)
- **Katakana**: カタカナ (foreign words)  
- **Kanji**: 漢字 (Chinese characters)
- **Mixed scripts**: ひらがなとKatakanaと漢字

## 🚨 Troubleshooting

**Voice profile not found:**
```bash
# Set up voice profile first
python3 easy_korean_tts.py --setup_voice
```

**Model loading issues:**
- Make sure you have sufficient RAM (8GB+ recommended)
- Check internet connection for model download
- Try different device: `--device cpu`

**Audio format errors:**
- Ensure reference.wav is proper WAV format
- Check file permissions in my_voice/ directory

## 🎌 Cultural Notes

When using Japanese TTS:
- Formal speech (`desu/masu`) is generally safer
- Context matters for politeness levels
- Some romanizations have multiple valid forms
- Regional variations exist in pronunciation

## 📚 Learning Resources

Use this TTS to practice:
- Pronunciation of common phrases
- Listening to your own voice in Japanese
- Creating study materials
- Building confidence in speaking

The Japanese TTS system is perfect for language learners who want to hear how Japanese sounds in their own voice! 