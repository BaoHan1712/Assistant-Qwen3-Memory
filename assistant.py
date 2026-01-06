import sounddevice as sd
import numpy as np
import soundfile as sf
import tempfile
import os
import subprocess
import time
import re
from difflib import SequenceMatcher

from faster_whisper import WhisperModel
from ollama import Client
from text_processor import process_ai_response, sanitize_user_input
from memory import ConversationMemory

# ================== CONFIG ==================
SAMPLE_RATE = 16000

WAKE_SECONDS   = 3
RECORD_SECONDS = 6

WAKE_WORD = "Bảo ơi"

OLLAMA_MODEL = "gemma3:1b"

BEEP_START = "assets/bip.wav"
BEEP_STOP  = "assets/bip2.wav"

PIPER_EXE  = r"piper.exe"
TTS_MODEL  = r"assets\vi_VN-vais1000-medium.onnx"
TTS_OUT    = r"assets\answer.wav"
# ===========================================


# ---------- Ollama ----------
client = Client(host="http://localhost:11434")

# ---------- Memory ----------
memory = ConversationMemory("conversation_memory.json", max_history=20)

# ---------- STT ----------
print("⏳ Loading Whisper STT model (Vietnamese)...")
stt_model = WhisperModel(
    "medium",          # tốt cho tiếng Việt
    device="cuda",
    compute_type="float16"
)
print("✅ STT loaded")


# ================== UTILS ==================
def remove_vietnamese_accents(text: str) -> str:
    text = text.lower()
    replacements = {
        "àáạảãâầấậẩẫăằắặẳẵ": "a",
        "èéẹẻẽêềếệểễ": "e",
        "ìíịỉĩ": "i",
        "òóọỏõôồốộổỗơờớợởỡ": "o",
        "ùúụủũưừứựửữ": "u",
        "ỳýỵỷỹ": "y",
        "đ": "d",
    }
    for chars, rep in replacements.items():
        for c in chars:
            text = text.replace(c, rep)
    return text


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


# ================== AUDIO ==================
def play_sound(path):
    if not os.path.exists(path):
        return
    data, sr = sf.read(path, dtype="float32")
    sd.play(data, sr)
    sd.wait()


def record_audio(seconds):
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.float32
    )
    sd.wait()
    return audio.flatten()


# ================== STT ==================
def speech_to_text(audio):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, SAMPLE_RATE)
        wav_path = f.name

    segments, _ = stt_model.transcribe(wav_path, language="vi")
    os.remove(wav_path)

    text = "".join(seg.text for seg in segments).strip()
    return sanitize_user_input(text)


# ================== WAKE WORD ==================
def detect_wake_word(text: str) -> bool:
    if not text:
        return False

    raw = text.lower()
    norm = remove_vietnamese_accents(raw)

    # Cách 1: chứa từ khoá
    if "bao" in norm and "oi" in norm:
        return True

    # Cách 2: fuzzy matching
    score = similarity(norm, remove_vietnamese_accents(WAKE_WORD))
    return score >= 0.4   # chỉnh 0.5–0.7 tuỳ môi trường

# hàm chờ đánh thức bằng wake word
def wait_for_wake_word():
    print(f"🟡 Chờ wake word: '{WAKE_WORD}' ...")
    audio = record_audio(WAKE_SECONDS)
    text = speech_to_text(audio)

    if text:
        print("👂 Nghe:", text)
        return detect_wake_word(text)

    return False


# ================== LLM ==================
def ask_ollama(text):
    memory.add_message("user", text)
    
    # System prompt với rules
    system_prompt = """Bạn là Bảo, một trợ lý hỗ trợ được tạo ra bởi Hàn Bảo.

RULES:
- Tên của bạn là Bảo
- Bạn được tạo ra bởi Hàn Bảo
- CHỈ trả lời bằng TIẾNG VIỆT, không dùng ngôn ngữ khác
- Luôn thân thiện, hỗ trợ người dùng
- Nếu được hỏi bằng tiếng khác, hãy trả lời tiếng Việt, không dùng icons hay emoji. Viết đoạn văn không có ký hiệu đặt biệt, gạch đầu dòng, hay định dạng markdown."""
    
    # Xây dựng messages với system prompt
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(memory.get_context())
    
    response = client.chat(
        model=OLLAMA_MODEL,
        messages=messages
    )

    answer = response["message"]["content"].strip()
    answer = process_ai_response(answer, max_chars=500)

    memory.add_message("assistant", answer)
    print("🤖 Bảo:", answer)
    return answer


# ================== TTS ==================
def text_to_speech(text):
    subprocess.run(
        [
            PIPER_EXE,
            "--model", TTS_MODEL,
            text,
            "--output-file", TTS_OUT
        ],
        check=True
    )
    data, sr = sf.read(TTS_OUT, dtype="float32")
    sd.play(data, sr)
    sd.wait()


# ================== MAIN ==================
def main_loop():
    print("\n🟢 Voice assistant started (wake-word fuzzy mode)\n")

    while True:
        try:
            if not wait_for_wake_word():
                continue

            print("🟢 Wake word detected!")
            play_sound(BEEP_START)

            audio = record_audio(RECORD_SECONDS)
            play_sound(BEEP_STOP)

            text = speech_to_text(audio)
            if not text:
                continue

            print("📝 User:", text)

            answer = ask_ollama(text)
            if answer:
                text_to_speech(answer)

            print("-" * 50)

        except KeyboardInterrupt:
            print("\n🛑 Exit")
            break

        except Exception as e:
            print("❌ Error:", e)
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main_loop()
