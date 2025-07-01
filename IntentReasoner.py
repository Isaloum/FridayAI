# ======================================
# File: IntentReasoner.py
# Purpose: Enables FridayAI to infer subtextual intent behind user input using semantics, emotion, and tone analysis
# ======================================

import re
from typing import Dict, Any
from transformers import pipeline

# === Initialize zero-shot classification model ===
# Note: You may swap this with a local fine-tuned model later
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


class IntentReasoner:
    # Detects underlying user intent categories like sarcasm, dismissal, avoidance, etc.
    def __init__(self):
        self.intent_labels = [
            "sarcasm",
            "dismissal",
            "emotional avoidance",
            "passive aggression",
            "empathy",
            "agreement",
            "genuine curiosity",
            "emotional vulnerability"
        ]

    def analyze_intent(self, text: str, emotion_hint: str = "neutral") -> Dict[str, Any]:
        cleaned = re.sub(r"[^\w\s\'\,\.!?]", "", text).strip()

        result = classifier(cleaned, self.intent_labels)
        scores = dict(zip(result['labels'], result['scores']))

        top_intent = result['labels'][0]
        confidence = result['scores'][0]

        needs_reflection = top_intent in ["emotional avoidance", "passive aggression", "sarcasm"]

        return {
            "intent": top_intent,
            "confidence": confidence,
            "needs_reflection": needs_reflection,
            "emotion_hint": emotion_hint,
            "raw_scores": scores
        }


# === EXAMPLE USAGE ===
if __name__ == "__main__":
    reasoner = IntentReasoner()

    samples = [
        "Fine. Whatever.",
        "No worries, I'm totally cool.",
        "I just… don’t want to talk about it.",
        "Wow, great idea. Love it."
    ]

    for s in samples:
        res = reasoner.analyze_intent(s)
        print(f"\nInput: {s}\n→ Intent: {res['intent']} ({res['confidence']:.2f}) | Reflection Needed: {res['needs_reflection']}")
