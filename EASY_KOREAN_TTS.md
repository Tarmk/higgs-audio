# 🇰🇷 Easy Korean TTS - No Korean Reading Required!

Generate Korean speech in **your voice** without needing to read Korean! Perfect if you want to say Korean phrases but can't read Hangul.

## 🚀 Quick Start (3 Steps)

### 1. Record Your Voice (in English)
```bash
python3 easy_korean_tts.py --setup_voice
```
- You'll read **English text** (which you can obviously read!)
- Takes 10-15 seconds
- This captures your voice characteristics

### 2. Generate Korean Speech
**Try a simple greeting:**
```bash
python3 easy_korean_tts.py --romanized "annyeonghaseyo"
```

**Interactive mode (recommended):**
```bash
python3 easy_korean_tts.py
```

## 🎯 How It Works

1. **Record in English** → Your voice profile is saved
2. **Type romanized Korean** → Like "annyeonghaseyo" 
3. **Get Korean audio** → In your voice!

## 📝 Common Korean Phrases You Can Try

Just type these romanized versions:

| Type this | You get | Meaning |
|-----------|---------|---------|
| `annyeonghaseyo` | 안녕하세요 | Hello |
| `gamsahamnida` | 감사합니다 | Thank you |
| `saranghae` | 사랑해 | I love you |
| `jal jayo` | 잘 자요 | Good night |
| `annyeonghi gaseyo` | 안녕히 가세요 | Goodbye |

## 🎮 Examples

**See all available phrases:**
```bash
python3 easy_korean_tts.py --phrases
```

**Generate specific phrases:**
```bash
# Hello
python3 easy_korean_tts.py --romanized "annyeonghaseyo"

# Thank you  
python3 easy_korean_tts.py --romanized "gamsahamnida"

# I love you
python3 easy_korean_tts.py --romanized "saranghae"
```

**Interactive mode:**
```bash
python3 easy_korean_tts.py

# Then type:
annyeonghaseyo
gamsahamnida  
saranghae
# etc...
```

## 💡 Pro Tips

1. **Record good quality English audio** - this affects all Korean speech quality
2. **Try different seeds** for variation: `--seed 123`
3. **You can also paste Korean text** if someone gives you Korean to say
4. **Type 'phrases'** in interactive mode to see all available phrases

## 🔧 Setup Your Voice

The setup is super easy:

1. Run: `python3 easy_korean_tts.py --setup_voice`
2. Read the English text it shows you
3. Record yourself (10-15 seconds)
4. Save as `my_voice/reference.wav`
5. Done! Now generate Korean speech

## 🎵 What You Get

- High-quality WAV files (24kHz)
- Korean speech in YOUR voice
- Works with any audio player
- Perfect for learning Korean pronunciation

---

**Try it now:**
```bash
python3 easy_korean_tts.py --setup_voice
python3 easy_korean_tts.py --romanized "annyeonghaseyo"
```

Easy Korean TTS - Say it in Korean, sound like yourself! 🎉 