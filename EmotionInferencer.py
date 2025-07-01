# =====================================
# EmotionInferencer.py – Psychological Intent Decoder
# =====================================

from typing import Dict

class EmotionInferencer:
    """
    Decodes a tone vector and emotional profile into psychological insights,
    empathy strategies, and behavioral flags.
    """

    def __init__(self):
        # Future-proof: can load model weights, rulesets, or RL profiles here
        self.rules = self._load_rules()

    def _load_rules(self):
        """Initialize default mappings for primary emotion patterns"""
        return {
            "sadness": {
                "low": {"style": "check_in", "depth": "light", "risk": "low"},
                "medium": {"style": "gentle_reflection", "depth": "deep", "risk": "medium"},
                "high": {"style": "hold_space", "depth": "very_deep", "risk": "high"}
            },
            "anger": {
                "low": {"style": "validating", "depth": "light", "risk": "low"},
                "medium": {"style": "firm_support", "depth": "medium", "risk": "medium"},
                "high": {"style": "de-escalate", "depth": "cautious", "risk": "high"}
            },
            "joy": {
                "any": {"style": "celebrate", "depth": "light", "risk": "none"}
            },
            "neutral": {
                "any": {"style": "engage_normal", "depth": "light", "risk": "none"}
            },
            "fear": {
                "medium": {"style": "reassure", "depth": "deep", "risk": "medium"},
                "high": {"style": "stabilize", "depth": "deep", "risk": "high"}
            }
        }

    def infer(self, tone_profile: Dict) -> Dict:
        """
        Infers empathy strategy, depth, and behavioral guidance
        based on tone vector, expression mode, and emotional certainty.
        """
        primary = tone_profile.get("primary", "unknown")
        certainty = tone_profile.get("certainty", 0.0)
        expression = tone_profile.get("expression_mode", "direct")
        vector = tone_profile.get("vector", {})

        # Quantize certainty level
        if certainty > 0.75:
            level = "high"
        elif certainty > 0.45:
            level = "medium"
        else:
            level = "low"

        # Choose rule mapping
        rule = self.rules.get(primary, {}).get(level)
        if not rule:
            rule = self.rules.get(primary, {}).get("any") or {
                "style": "reflective", "depth": "light", "risk": "unknown"
            }

        # Add behavioral flags based on expression masking
        if expression == "masked" and rule["risk"] != "none":
            behavior = "suppressing_pain"
        elif expression == "direct" and rule["risk"] == "high":
            behavior = "crisis_disclosure"
        else:
            behavior = "open_state"

        return {
            "empathy_style": rule["style"],
            "engagement_depth": rule["depth"],
            "risk_level": rule["risk"],
            "user_state": behavior,
            "certainty": certainty,
            "primary_emotion": primary,
            "expression_mode": expression
        }


# =====================
# CLI Test Mode
# =====================
if __name__ == "__main__":
    print("\n🧠 EmotionInferencer Test Console")

    test_sample = {
        "primary": "sadness",
        "secondary": "anger",
        "certainty": 0.81,
        "expression_mode": "masked",
        "vector": {
            "sadness": 0.81,
            "joy": 0.05,
            "neutral": 0.12
        }
    }

    engine = EmotionInferencer()
    result = engine.infer(test_sample)

    print("\n🧩 Inferred Psychological Profile:")
    for k, v in result.items():
        print(f"{k}: {v}")
    print("-" * 40)
