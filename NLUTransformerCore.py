# ======================================
# File: NLUTransformerCore.py
# Purpose: Preprocess user input for FridayAI by handling slang, typos,
#          repeated characters, and detecting language for robust understanding.
# ======================================

import re
from typing import Tuple
from langdetect import detect
from SpellCorrectionCore import SpellCorrectionCore

class NLUTransformerCore:
    """
    Preprocesses user input for robust Natural Language Understanding (NLU).
    Handles:
    - Repetition cleanup (e.g., "heyyyyy" → "hey")
    - Slang/shortcut expansion ("wyd" → "what are you doing")
    - Spell correction using SpellCorrectionCore
    - Language detection (using langdetect)
    """

    def __init__(self):
        # Mapping of common slang/shortcuts to full phrases
        self.slang_map = {
            "brb": "be right back",
            "lol": "laughing out loud",
            "idk": "I don't know",
            "tbh": "to be honest",
            "omg": "oh my god",
            "wya": "where are you",
            "smh": "shaking my head",
            "wyd": "what are you doing",
            "imo": "in my opinion",
            "btw": "by the way",
            "rn": "right now",
            "u": "you",
            "r": "are",
            "gonna": "going to",
            "wanna": "want to"
        }
        self.spell_corrector = SpellCorrectionCore()  # 🔍 Injected spell corrector

    def process(self, text: str) -> Tuple[str, str]:
        """
        Normalize input by:
        1. Reducing character repetition
        2. Expanding slang
        3. Spell correction
        4. Detecting language
        Returns tuple of (cleaned_text, language_code)
        """
        original = text
        text = self.reduce_repetitions(text)        # Normalize elongated words
        text = self.expand_slang(text)              # Expand known shortcuts
        text = self.spell_corrector.correct_sentence(text)  # Apply spelling correction
        lang = self.detect_language(text)           # Detect input language
        return text, lang

    def reduce_repetitions(self, text: str) -> str:
        """
        Collapse repeated characters (e.g., "heyyyyy" → "hey")
        Helps reduce noise from informal typing.
        """
        return re.sub(r'(.)\1{2,}', r'\1', text)

    def expand_slang(self, text: str) -> str:
        """
        Replace known slang/shortcut words with full phrases.
        Uses self.slang_map defined in __init__.
        """
        words = text.lower().split()
        return ' '.join(self.slang_map.get(word, word) for word in words)

    def detect_language(self, text: str) -> str:
        """
        Detect the language of the input using langdetect.
        Returns 2-letter language code (e.g., 'en' for English).
        If detection fails, returns 'unknown'.
        """
        try:
            return detect(text)
        except:
            return "unknown"


# ==========================
# Quick CLI Test (offline)
# ==========================
if __name__ == "__main__":
    nlu = NLUTransformerCore()
    while True:
        try:
            raw = input("You: ").strip()
            if raw.lower() in ["exit", "quit"]:
                break
            processed, lang = nlu.process(raw)
            print(f"\n🔍 Processed: {processed}\n🌍 Language: {lang}\n")
        except KeyboardInterrupt:
            break
