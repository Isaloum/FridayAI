# ==============================================
# File: core/EmotionClassifier.py
# Purpose: NLP-based emotion detector using HuggingFace transformer
# ==============================================

from transformers import pipeline
from typing import Dict

class EmotionClassifier:
    """
    Classifies free-form text into emotional categories.
    """
    def __init__(self):
        # 🤗 Load HuggingFace emotion classification pipeline
        self.classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None
        )

    def analyze(self, text: str) -> Dict:
        import re

        # ✅ Basic word filter for nonsense detection
        def is_gibberish(text: str) -> bool:
            text = text.strip().lower()
            if len(text) < 3:
                return True
            if not re.search(r'[a-zA-Z]', text):  # no letters
                return True
            if not re.search(r'[aeiou]', text):  # no vowels
                return True
            if re.fullmatch(r'[a-z]{4,}', text) and text not in COMMON_WORDS:
                return True
            return False

        COMMON_WORDS = {
            "happy", "sad", "tired", "okay", "worried",
            "joy", "angry", "good", "bad", "calm", "great"
        }

        # ❌ Abort early if gibberish
        if is_gibberish(text):
            return {
                "top_emotion": "unknown",
                "certainty": 0.0,
                "vector": {},
                "expression_mode": "noise"
            }

        try:
            results = self.classifier(text)[0]  # list of dicts
            sorted_results = sorted(results, key=lambda r: r['score'], reverse=True)
            top = sorted_results[0]

            # ❌ Abort if model isn't confident
            if top['score'] < 0.5:
                return {
                    "top_emotion": "unknown",
                    "certainty": round(top['score'], 4),
                    "vector": {},
                    "expression_mode": "uncertain"
                }

            # ✅ Return clean emotional profile
            return {
                "top_emotion": top['label'].lower(),
                "certainty": round(top['score'], 4),
                "vector": {r['label'].lower(): r['score'] for r in sorted_results}
            }

        except Exception as e:
            return {
                "top_emotion": "unknown",
                "certainty": 0.0,
                "vector": {},
                "error": str(e)
            }


# =====================
# CLI Test Mode
# =====================
if __name__ == "__main__":
    ec = EmotionClassifier()
    print("\n🔍 EmotionClassifier Test Mode")
    while True:
        text = input("You: ").strip()
        if text.lower() in ["exit", "quit"]:
            break
        profile = ec.analyze(text)
        print(f"\nTop Emotion: {profile['top_emotion']} ({profile['certainty']*100:.1f}%)")
        print(f"Full Vector: {profile['vector']}")
        print("-" * 40)
