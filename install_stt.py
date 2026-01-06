from faster_whisper import WhisperModel

print("⏳ Loading STT model...")
stt_model = WhisperModel(
    "medium",
    device="cpu",
    compute_type="int8"
)
print("✅ STT loaded")