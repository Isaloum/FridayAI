# =====================================
# ToneVectorizer.py – Emotional Signature Encoder (Core Layer)
# =====================================

from typing import Dict
from transformers import pipeline
import torch
from core.EmotionLayerCore import EmotionLayerCore

class ToneVectorizer:
    """
    Transforms user input into emotional signature vectors
    with offline fallback using keyword detection.
    """

    def __init__(self):
        # Try load transformer model (online)
        try:
            self.classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
        except Exception as e:
            self.classifier = None
        # Load keyword fallback (offline)
        self.keyword_detector = EmotionLayerCore()

    def encode(self, text: str) -> Dict:
        # Try transformer first
        if self.classifier:
            try:
                results = self.classifier(text)[0]
                scores = {res['label'].lower(): res['score'] for res in results}
                sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)

                primary = sorted_emotions[0][0]
                secondary = sorted_emotions[1][0] if len(sorted_emotions) > 1 else None
                certainty = sorted_emotions[0][1]

                expression_mode = "masked" if primary in ["neutral", "joy"] and scores.get("sadness", 0) > 0.3 else "direct"

                # Low confidence fallback
                if certainty < 0.5:
                    return self._fallback_encode(text)

                return {
                    "vector": scores,
                    "primary": primary,
                    "secondary": secondary,
                    "expression_mode": expression_mode,
                    "certainty": round(certainty, 3)
                }

            except Exception:
                return self._fallback_encode(text)

        else:
            return self._fallback_encode(text)

    def _fallback_encode(self, text: str) -> Dict:
        # Use EmotionLayerCore when transformer fails
        result = self.keyword_detector.detect_emotion_layers(text)
        primary = result["primary"]
        secondary = result["supporting"][0] if result["supporting"] else None
        return {
            "vector": result["raw_scores"],
            "primary": primary,
            "secondary": secondary,
            "expression_mode": "keyword",
            "certainty": 0.4
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
