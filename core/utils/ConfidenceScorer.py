# ==============================================
# File: core/utils/ConfidenceScorer.py
# Purpose: Heuristic-based confidence labeler for emotion reflections
# ==============================================

class ConfidenceScorer:
    @staticmethod
    def label_confidence(tone_analysis: dict, layer_analysis: dict) -> str:
        """
        Estimate confidence based on tone and emotional layer data.
        Returns: 'high', 'medium', or 'low'
        """
        tone_certainty = tone_analysis.get("certainty", 0.0)
        primary_tone = tone_analysis.get("primary", "neutral")
        primary_layer = layer_analysis.get("primary", "neutral")

        supporting = layer_analysis.get("supporting", [])
        alignment = (primary_tone == primary_layer)
        multi_layer = len(supporting) >= 1

        # Basic heuristics
        if tone_certainty > 0.85 and alignment:
            return "high"
        elif tone_certainty > 0.5 and (alignment or multi_layer):
            return "medium"
        return "low"
