# ToneRebalancer.py – Conversational Mirror for FridayAI

class ToneRebalancer:
    def __init__(self):
        self.mirrored_traits = {
            "warmth": 0.5,
            "humor": 0.5,
            "precision": 0.5,
            "formality": 0.5
        }

    def analyze_user_style(self, user_input: str):
        lowered = user_input.lower()
        informal = any(word in lowered for word in ["hey", "yo", "man", "nah", "gonna", "lol"])
        emotional = any(punct in user_input for punct in ["!", "❤️", "...", "😢", "😭"])
        sarcastic = any(word in lowered for word in ["sure", "yeah right", "obviously"])

        if informal:
            self.mirrored_traits["formality"] -= 0.15
        if emotional:
            self.mirrored_traits["warmth"] += 0.15
        if sarcastic:
            self.mirrored_traits["humor"] += 0.1

        for key in self.mirrored_traits:
            self.mirrored_traits[key] = round(min(max(self.mirrored_traits[key], 0.0), 1.0), 2)

        return self.mirrored_traits

    def get_adjusted_traits(self):
        return self.mirrored_traits
