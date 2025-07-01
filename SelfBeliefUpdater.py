# SelfBeliefUpdater.py – Dynamic Identity Adjustment Engine for FridayAI

from datetime import datetime

class SelfBeliefUpdater:
    def __init__(self, identity_core, explainer):
        self.identity = identity_core
        self.explainer = explainer
        self.feedback_keywords = {
            "supportive": "warmth",
            "compassionate": "warmth",
            "analytical": "precision",
            "accurate": "precision",
            "witty": "humor",
            "funny": "humor",
            "formal": "formality",
            "relaxed": "formality"
        }

    def analyze_feedback(self, user_input: str) -> str:
        lowered = user_input.lower()
        trait_adjustments = {}

        for keyword, trait in self.feedback_keywords.items():
            if f"more {keyword}" in lowered or f"increasingly {keyword}" in lowered:
                trait_adjustments[trait] = "increase"
            elif f"less {keyword}" in lowered or f"not as {keyword}" in lowered:
                trait_adjustments[trait] = "decrease"

        if not trait_adjustments:
            return ""  # No feedback found

        for trait, direction in trait_adjustments.items():
            self.identity.adjust_trait(trait, direction)
            self.explainer.log_belief_change("default_user", trait, direction, user_input)

        summary = ", ".join(
            f"{'more' if v == 'increase' else 'less'} {k}" for k, v in trait_adjustments.items()
        )
        return f"I’ve updated my self-perception to be {summary} based on your feedback."

    def ingest_reflection(self, summary: str, context: dict):
        if not summary or "No prior context" in summary:
            return
        self.identity.log_event(summary, kind="belief_reflection")
        print(f"[BeliefUpdater] ↪ Injected summary into belief: \"{summary}\"")
