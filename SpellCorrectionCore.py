# ======================================
# File: SpellCorrectionCore.py
# Purpose: Lightweight spell correction module for FridayAI using pyspellchecker.
# ======================================

from spellchecker import SpellChecker

class SpellCorrectionCore:
    """
    Wraps the pyspellchecker to provide single-word and sentence-level correction.
    Can be swapped or extended without affecting Friday's architecture.
    """

    def __init__(self):
        self.spell = SpellChecker()

    def correct_word(self, word: str) -> str:
        """
        Returns the corrected version of a single word.
        Only corrects if the word is not already known.
        """
        return word if word in self.spell else self.spell.correction(word)

    def correct_sentence(self, sentence: str) -> str:
        """
        Applies correction to all words in a sentence.
        """
        words = sentence.split()
        corrected = [self.correct_word(word) for word in words]
        return ' '.join(corrected)


# =============================
# Test This Module (Offline)
# =============================
if __name__ == "__main__":
    corrector = SpellCorrectionCore()
    while True:
        try:
            text = input("You (with typos): ").strip()
            if text.lower() in ["exit", "quit"]:
                break

            corrected = corrector.correct_sentence(text)
            print(f"🔍 Corrected: {corrected}\n")
        except KeyboardInterrupt:
            break
