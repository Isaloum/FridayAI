# =====================================
# EmpathyReasoner.py – Unified Empathy Brain Layer 
# 🔥 Fused with ToneVectorizer, EmotionInferencer, EmpathySelector
# =====================================

import re
from datetime import datetime
from ToneVectorizer import ToneVectorizer
from EmotionInferencer import EmotionInferencer
from EmpathySelector import EmpathySelector
from MemoryCore import MemoryCore
from EmotionCore import EmotionCore

# =====================================
# Function: estimate_emotional_weight
# Purpose: Compute emotional intensity score for memory salience
# =====================================
def estimate_emotional_weight(text: str, primary_emotion: str) -> float:
    """
    Estimate emotional salience based on emotion type and critical phrases.
    Used for memory weighting and reflective prioritization.
    """
    base_weights = {
        "joy": 0.3,
        "neutral": 0.1,
        "surprise": 0.4,
        "sadness": 0.8,
        "anger": 0.85,
        "fear": 0.9,
        "disgust": 0.75
    }

    emotion_weight = base_weights.get(primary_emotion.lower(), 0.2)

    # High-intensity emotional triggers
    intensity_keywords = [
        r"\bbroken\b", r"\blost\b", r"\bworthless\b", r"\bhopeless\b",
        r"\bcrisis\b", r"\bpanic\b", r"\bdon't care\b", r"\bwhat's the point\b",
        r"\bempty\b", r"\balone\b", r"\bgive up\b"
    ]

    multiplier = 1.0
    for kw in intensity_keywords:
        if re.search(kw, text, re.IGNORECASE):
            multiplier += 0.1

    return round(min(emotion_weight * multiplier, 1.0), 2)


# =====================================
# Class: EmpathyReasoner
# Purpose: Process tone, infer intent, suggest empathy strategy
# =====================================
class EmpathyReasoner:
    """
    Central empathy engine – detects emotional nuance,
    infers psychological intent, and selects adaptive response strategy.
    """

    def __init__(self, emotion_core: EmotionCore, memory_core: MemoryCore):
        self.emotion = emotion_core
        self.memory = memory_core
        self.vectorizer = ToneVectorizer()
        self.inferencer = EmotionInferencer()
        self.selector = EmpathySelector()

    def analyze_subtext(self, user_input: str) -> dict:
        """
        Full empathy pipeline: analyze tone, infer intent, choose response style.
        Returns structured empathy cue for Friday's dialogue engine.
        """
        try:
            # Step 1: Emotional encoding
            tone_profile = self.vectorizer.encode(user_input)

            # Step 2: Infer psychological state
            inferred = self.inferencer.infer(tone_profile)

            # Step 3: Select strategic response template
            strategy = self.selector.select(inferred)

            # Step 4: Log high-emotion input into memory core
            if inferred['risk_level'] in ["high", "medium"]:
                self.memory.flag_important_memory({
                    "type": "empathy_trigger",
                    "subtext_flags": [inferred["primary_emotion"]],
                    "original_text": user_input,
                    "timestamp": datetime.now().isoformat(),
                    "suggested_response": strategy["opening_line"]
                })

            # Step 5: Return structured empathy metadata
            return {
                "empathy_cue": strategy["strategy_tag"],
                "empathy_reply": strategy["opening_line"],
                "tone": tone_profile,
                "inferred": inferred,
                "strategy": strategy
            }

        except Exception as e:
            return {
                "empathy_cue": None,
                "error": str(e),
                "fallback": "I’m here if you want to talk about it."
            }


# =====================
# CLI Test Mode
# =====================
if __name__ == "__main__":
    print("\n🧠 EmpathyReasoner Unified Test Console")
    er = EmpathyReasoner(EmotionCore(), MemoryCore())

    while True:
        try:
            text = input("You: ").strip()
            if text.lower() in ["exit", "quit"]:
                break

            result = er.analyze_subtext(text)

            print("\n🎯 Empathy Analysis Result")
            print(f"Empathy Cue: {result['empathy_cue']}")
            print(f"Reply: {result['empathy_reply']}")
            print(f"Tone: {result['tone']}")
            print(f"Inferred: {result['inferred']}")
            print(f"Strategy: {result['strategy']}")
            print("-" * 40)

        except KeyboardInterrupt:
            print("\nSession ended.")
            break
