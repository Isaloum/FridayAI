# ======================================
# File: UserSlangTrainer.py
# Purpose: Allows FridayAI to learn user-specific slang from missed_words.log and evolve custom_words.txt adaptively
# ======================================

import os
import json
from typing import List

# === CONFIGURATION ===
MISSED_WORDS_FILE = "missed_words.log"
CUSTOM_WORDS_FILE = "custom_words.txt"
APPROVED_WORDS_FILE = "approved_slang.json"


class UserSlangTrainer:
    # Allows interactive slang approval and dictionary expansion
    def __init__(self):
        self.missed_words = self._load_missed_words()
        self.custom_words = self._load_custom_words()
        self.approved_words = self._load_approved_words()

    def _load_missed_words(self) -> List[str]:
        if not os.path.exists(MISSED_WORDS_FILE):
            return []
        with open(MISSED_WORDS_FILE, "r", encoding="utf-8") as f:
            return list(sorted(set(line.strip() for line in f if line.strip())))

    def _load_custom_words(self) -> List[str]:
        if not os.path.exists(CUSTOM_WORDS_FILE):
            return []
        with open(CUSTOM_WORDS_FILE, "r", encoding="utf-8") as f:
            return list(sorted(set(line.strip() for line in f if line.strip())))

    def _load_approved_words(self) -> dict:
        if not os.path.exists(APPROVED_WORDS_FILE):
            return {}
        with open(APPROVED_WORDS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_custom_words(self):
        all_words = set(self.custom_words + list(self.approved_words.keys()))
        with open(CUSTOM_WORDS_FILE, "w", encoding="utf-8") as f:
            for word in sorted(all_words):
                f.write(word + "\n")

    def _save_approved_words(self):
        with open(APPROVED_WORDS_FILE, "w", encoding="utf-8") as f:
            json.dump(self.approved_words, f, indent=2)

    def review_and_train(self):
        print("\n=== 🧠 FridayAI Slang Trainer ===")
        for word in self.missed_words:
            if word in self.approved_words:
                continue

            print(f"\nNew slang detected: '{word}'")
            meaning = input("Define this slang (or leave blank to skip): ").strip()
            if not meaning:
                continue

            self.approved_words[word] = meaning

        self._save_approved_words()
        self._save_custom_words()
        print("\n✅ Slang training complete. custom_words.txt updated.")


# === EXAMPLE USAGE ===
if __name__ == "__main__":
    trainer = UserSlangTrainer()
    trainer.review_and_train()

    # Example: User defines 'prolly' as 'probably', adds to custom_words.txt
