# ======================================
# File: InputMemoryLogger.py
# Purpose: Logs cleaned user input, normalized text, and system responses for future memory/fine-tuning.
# ======================================

import os
import json
from datetime import datetime

CORPUS_FILE = os.path.join("assets", "FridayCorpus.jsonl")

class InputMemoryLogger:
    def __init__(self, path=CORPUS_FILE):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path

    def log(self, raw_input: str, cleaned: str, response: str, metadata: dict = None):
        record = {
            "timestamp": datetime.now().isoformat(),
            "input": raw_input,
            "normalized": cleaned,
            "response": response,
            "meta": metadata or {}
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


# === EXAMPLE USAGE ===
if __name__ == "__main__":
    logger = InputMemoryLogger()
    logger.log(
        raw_input="yo bruh she wildin rn",
        cleaned="She is acting wild right now",
        response="Do you want to talk about it?",
        metadata={"emotion": "neutral", "domain": "default_chat"}
    )
    print("✅ Logged interaction to FridayCorpus.jsonl")
