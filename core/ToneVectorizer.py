# =====================================
# ToneVectorizer.py – Emotional Signature Encoder (Core Layer)
# =====================================

from typing import Dict
from transformers import pipeline
import torch

class ToneVectorizer:
    """
    Transforms user input into emotional signature vectors
    for deep empathy, context mirroring, and response adaptation.
    """

    def __init__(self):
        # 🧠 Load pre-trained multilingual emotion classifier
        try:
            self.classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
        except Exception as e:
            raise RuntimeError(f"Failed to load ToneVectorizer model: {e}")

    def encode(self, text: str) -> Dict:
        """
        Process the input sentence and return an emotional tone profile.

        Returns:
        - emotional_vector: raw label → probability map
        - primary: top detected emotion
        - secondary: next strong emotion
        - expression_mode: basic inference of emotional masking
        - certainty: how dominant the primary emotion is
        """
        try:
            results = self.classifier(text)[0]  # Get top label-probability pairs
            scores = {res['label'].lower(): res['score'] for res in results}
            sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)

            primary = sorted_emotions[0][0]
            secondary = sorted_emotions[1][0] if len(sorted_emotions) > 1 else None
            certainty = sorted_emotions[0][1]

            # 🌫️ Simple masking detection heuristic
            expression_mode = "masked" if primary in ["neutral", "joy"] and scores.get("sadness", 0) > 0.3 else "direct"

            return {
                "vector": scores,
                "primary": primary,
                "secondary": secondary,
                "expression_mode": expression_mode,
                "certainty": round(certainty, 3)
            }

        except Exception as e:
            return {
                "vector": {},
                "primary": "unknown",
                "secondary": None,
                "expression_mode": "uncertain",
                "certainty": 0.0,
                "error": str(e)
            }


# =====================
# CLI Test Example
# =====================
if __name__ == "__main__":
    print("\n🧠 ToneVectorizer Test Mode")
    tv = ToneVectorizer()

    while True:
        try:
            text = input("You: ").strip()
            if text.lower() in ["exit", "quit"]:
                break

            profile = tv.encode(text)
            print("\n🎯 Tone Profile:")
            print(f"Primary: {profile['primary']}, Certainty: {profile['certainty']*100:.1f}%")
            print(f"Secondary: {profile['secondary']}")
            print(f"Expression Mode: {profile['expression_mode']}")
            print(f"Full Vector: {profile['vector']}")
            print("-" * 40)

        except KeyboardInterrupt:
            print("\nSession ended.")
            break
