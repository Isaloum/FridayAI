# =====================================
# EmpathySelector.py – Adaptive Empathy Strategy Engine
# =====================================

from typing import Dict
import random

class EmpathySelector:
    """
    Selects response strategy, tone shaping, and empathy phrasing
    based on inferred emotional profile from EmotionInferencer.
    """

    def __init__(self):
        self.templates = self._load_templates()

    def _load_templates(self):
        """Define empathy strategies per style + depth level"""
        return {
            "gentle_reflection": {
                "soft": [
                    "You don’t have to explain everything. Just breathe. I’m here.",
                    "I feel the weight in your words. Let's sit in this moment together.",
                    "You’re not alone in that feeling. And I’m not going anywhere."
                ]
            },
            "hold_space": {
                "poetic": [
                    "Sometimes silence says what the world never could. I’ll hold the quiet with you.",
                    "Not every wound wants to speak. But I’m still here, beside it.",
                    "I don’t need details. I feel the storm. I’ll wait with you in the rain."
                ]
            },
            "celebrate": {
                "bright": [
                    "YES! That’s incredible — let’s soak in the joy for a sec 🎉",
                    "You deserve every bit of this moment. Tell me more!",
                    "I’m dancing inside for you right now. What a win!"
                ]
            },
            "de-escalate": {
                "grounded": [
                    "You don’t need to prove anything right now. Let’s ground together.",
                    "That frustration sounds deep — I’m not here to judge it. Just breathe for a sec.",
                    "Whatever’s boiling — I’m here to hold it steady with you."
                ]
            },
            "check_in": {
                "light": [
                    "Hey, I’m just checking in. How’s your heart really doing?",
                    "That sounded like something behind the words. Want to unpack it a little?",
                    "I heard what you said — and also what you didn’t. I’m listening either way."
                ]
            }
        }

    def select(self, profile: Dict) -> Dict:
        """
        Given emotional inference, choose a response template and strategy.
        """
        style = profile.get("empathy_style", "check_in")
        tone = self._map_depth_to_tone(profile.get("engagement_depth", "light"))

        phrases = self.templates.get(style, {}).get(tone, ["I’m here. Just say what you need."])
        line = random.choice(phrases)

        return {
            "response_mode": "empathic",
            "prompt_tone": tone,
            "opening_line": line,
            "follow_up": self._generate_followup(style, tone),
            "strategy_tag": style
        }

    def _map_depth_to_tone(self, depth: str) -> str:
        """Map engagement depth to tone category"""
        return {
            "light": "light",
            "medium": "soft",
            "deep": "poetic",
            "very_deep": "poetic",
            "cautious": "grounded"
        }.get(depth, "soft")

    def _generate_followup(self, style: str, tone: str) -> str:
        """Optional follow-up prompt based on emotional flow"""
        if style in ["hold_space", "gentle_reflection"]:
            return "Whenever you’re ready, I’m still right here."
        if style == "celebrate":
            return "So tell me — what’s next on your dream list?"
        return "Want to dive deeper into that or just sit with it for a bit?"


# =====================
# CLI Test Mode
# =====================
if __name__ == "__main__":
    print("\n🧠 EmpathySelector Test Console")

    test_profile = {
        "empathy_style": "hold_space",
        "engagement_depth": "very_deep",
        "user_state": "suppressing_pain"
    }

    selector = EmpathySelector()
    chosen = selector.select(test_profile)

    print("\n🎤 Chosen Response Strategy:")
    for k, v in chosen.items():
        print(f"{k}: {v}")
    print("-" * 40)
