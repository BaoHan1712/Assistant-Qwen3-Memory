"""
Semantic Command Handler
Nhận diện lệnh cùng nghĩa bằng embedding và gửi xuống ESP32
"""

from send_uart import *
import re
import numpy as np
from sentence_transformers import SentenceTransformer, util

# ================== COMMAND CODES ==================
COMMAND_CODES = {
    "forward": 0x01,
    "backward": 0x02,
    "left": 0x03,
    "right": 0x04,
    "stop": 0x05,
}

# ================== COMMAND EXAMPLES ==================
# Câu mẫu cùng nghĩa cho mỗi lệnh
COMMAND_SENTENCES = {
    "forward": [
        "tiến lên",
        "đi thẳng",
        "đi về phía trước",
        "chạy lên phía trước",
        "tiến lên phía trước",
    ],
    "backward": [
        "lùi lại",
        "đi lùi",
        "lùi về phía sau",
        "chạy lùi",
    ],
    "left": [
        "quay trái",
        "rẽ trái",
        "sang trái",
    ],
    "right": [
        "quay phải",
        "rẽ phải",
        "sang phải",
    ],
    "stop": [
        "dừng lại",
        "dừng",
        "đứng yên",
        "ngừng di chuyển",
    ],
}

# ================== MODEL ==================
print("[*] Loading semantic command model...")
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("[OK] Model loaded")

# Precompute embeddings
COMMAND_EMBEDDINGS = {}
for cmd, sentences in COMMAND_SENTENCES.items():
    COMMAND_EMBEDDINGS[cmd] = embedder.encode(sentences, normalize_embeddings=True)

# ================== HANDLER ==================
class CommandHandler:
    def __init__(self, port=UART_PORT, baudrate=UART_BAUD, threshold=0.55):
        self.esp = ESP32Commander(port, baudrate)
        self.threshold = threshold

    def extract_steps(self, text: str) -> int:
        match = re.search(r"(\d+)\s*(?:bước|step)", text)
        return int(match.group(1)) if match else 1

    def detect_command(self, text: str):
        """
        Dùng semantic similarity để nhận diện lệnh
        """
        text = text.lower().strip()
        text_emb = embedder.encode(text, normalize_embeddings=True)

        best_cmd = None
        best_score = 0.0

        for cmd, emb_list in COMMAND_EMBEDDINGS.items():
            scores = util.cos_sim(text_emb, emb_list)
            score = float(scores.max())
            if score > best_score:
                best_score = score
                best_cmd = cmd

        if best_score >= self.threshold:
            return best_cmd, best_score

        return None, best_score

    def execute(self, text: str) -> bool:
        steps = self.extract_steps(text)
        cmd, score = self.detect_command(text)

        if cmd is None:
            print("[WARNING] Khong hieu lenh: " + text)
            return False

        code = COMMAND_CODES[cmd]
        print("[EXECUTE] '" + cmd + "' | score=" + f"{score:.2f}" + " | steps=" + str(steps))

        for i in range(steps):
            self.esp.send(code)

        return True

    def close(self):
        self.esp.close()


# # ================== TEST ==================
# if __name__ == "__main__":
#     handler = CommandHandler()

#     tests = [
#         "tiến lên phía trước 3 bước",
#         "đi thẳng",
#         "chạy lên phía trước",
#         "rẽ trái",
#         "quay phải",
#         "đứng yên",
#     ]

#     for t in tests:
#         print("\nUSER:", t)
#         handler.execute(t)

#     handler.close()
