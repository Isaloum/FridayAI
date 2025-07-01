# ==================================================
# File: core/pregnancy/PregnancyConfidenceScorer.py
# Purpose: (Placeholder) Provides dummy confidence levels
# ==================================================

class ConfidenceScorer:
    @staticmethod
    def score(emotion: str, certainty: float) -> str:
        if certainty >= 0.8:
            return "✅ High confidence"
        elif certainty >= 0.5:
            return "⚠️ Moderate confidence"
        else:
            return "❓ Low confidence — take this with care."
