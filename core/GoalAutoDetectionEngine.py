# GoalAutoDetectionEngine.py
# -------------------------------------------------------
# Monitors user input for goal-like statements and adds them to LongTermIntentCore automatically.

from datetime import datetime
import uuid
import re

class GoalAutoDetectionEngine:
    def __init__(self, intent_core, emotion_core):
        self.intent_core = intent_core
        self.emotion_core = emotion_core
        self.patterns = [
            r"i want to (.+)",
            r"i need to (.+)",
            r"i have to (.+)",
            r"one day i will (.+)",
            r"i'm going to (.+)",
            r"i should (.+)"
        ]

    def scan_and_log(self, user_input: str, emotion_summary: str):
        lowered = user_input.lower()
        for pattern in self.patterns:
            match = re.search(pattern, lowered)
            if match:
                intent_phrase = match.group(1).strip().rstrip(".!?")
                intent_id = f"auto_{uuid.uuid4().hex[:8]}"
                self.intent_core.add_intent(
                    intent_id=intent_id,
                    description=intent_phrase.capitalize(),
                    priority="medium",
                    category="inferred",
                    emotion_tags=[emotion_summary, "auto_generated"]
                )
                return {
                    "detected": True,
                    "intent_id": intent_id,
                    "description": intent_phrase
                }
        return {"detected": False}
