# ==============================================
# File: core/pregnancy/PregnancyEmotionPlanner.py
# Purpose: Generate personalized self-care plans from emotion signals
# ==============================================

class PregnancyEmotionPlanner:
    @staticmethod
    def generate_plan(emotion: str) -> list:
        # Predefined emotional care plans
        plans = {
            "sadness": [
                "Write one honest line in your journal.",
                "Light a candle or play soft music.",
                "Let yourself cry if you need."
            ],
            "anxious": [
                "Take three slow deep breaths.",
                "Name the exact worry — then reframe it.",
                "Step outside for a few minutes."
            ],
            "joy": [
                "Celebrate with a voice memo about today.",
                "Share your joy with someone you trust.",
                "Lock in this memory — it matters."
            ],
            "anger": [
                "Pause — and move physically (walk, stretch).",
                "Vent safely (journal or talk to someone).",
                "Try to name the need under the anger."
            ],
            "tired": [
                "Close your eyes for 2 minutes, now.",
                "Cancel one non-essential task today.",
                "Hydrate — even just a glass of water helps."
            ],
            "neutral": [
                "You’re allowed to just be.",
                "Try a small stretch or walk.",
                "Check in — what do you *want* right now?"
            ]
        }

        return plans.get(emotion, [
            "Your feelings are valid. Let's take a soft next step.",
            "Would journaling or silence feel better?",
            "Whatever it is — you're not alone."
        ])
