# ============================
# EmpathyCore.py
# ============================
# Generates empathetic responses based on detected emotions

from typing import Dict

class EmpathyCore:
    """
    Generates empathetic responses based on a dictionary of emotion weights.
    Prioritizes top 1–3 emotions and blends their responses together.
    """

    def __init__(self):
        self.templates = {
            "joy": "That's awesome! I'm glad you’re feeling good!",
            "sadness": "I'm really sorry you're feeling that way. You're not alone in this.",
            "anger": "That's understandable — it's okay to feel angry sometimes.",
            "fear": "That sounds scary. I'm here with you. Want to talk more about it?",
            "love": "That's beautiful — your heart is in the right place.",
            "shame": "You're not worthless. You matter, no matter what your thoughts say.",
            "guilt": "Everyone makes mistakes. What matters is that you care.",
            "confidence": "Yes — I see your strength. You’ve got this.",
            "none": "I'm here if you want to share more."
        }

    def generate_response(self, emotion_weights: Dict[str, float]) -> str:
        """
        Given detected emotion weights, returns a blended empathetic response.
        """
        if not emotion_weights:
            return self.templates["none"]

        # Sort emotions by weight descending and take top 3
        top_emotions = sorted(emotion_weights.items(), key=lambda x: -x[1])[:3]

        responses = [self.templates.get(emotion, "") for emotion, weight in top_emotions if weight > 0]
        return " ".join(responses).strip() or self.templates["none"]


# ============================
# Test Demo
# ============================
if __name__ == "__main__":
    empathy = EmpathyCore()

    test_emotions = {
        "joy": 0.6,
        "love": 0.3,
        "sadness": 0.1
    }

    print("🧠 Emotions Detected:", test_emotions)
    print("🤖 Friday:", empathy.generate_response(test_emotions))
