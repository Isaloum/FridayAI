# =============================================
# File: SelfAwarenessEngine.py
# Purpose: Generate introspective reflections and detect cognitive drift
# =============================================

from datetime import datetime

class SelfAwarenessEngine:
    """
    Synthesizes behavioral reflection from logs and tone history.
    Used in Option B reflection loop for narrative introspection.
    """

    def __init__(self, emotion_core, memory_core):
        self.emotion_core = emotion_core
        self.memory_core = memory_core
        self.last_drift_check = None

    def generate_self_reflection(self) -> str:
        """
        Analyze recent tone/memory logs and produce self-reflection text.
        """
        mood_trend = self.emotion_core.get_recent_mood_trend()
        reflections = self.memory_core.query_memories(
            filter_tags=["reflection"],
            since_hours=24
        )

        if not reflections:
            return "I don't have enough recent reflective data to generate a self-observation."

        mood_summary = f"Recent mood trend: {mood_trend}" if mood_trend else "Mood data unclear."
        memory_snippets = "\n".join([r['content'] for r in reflections[-3:]])

        return f"In the past 24 hours, my emotional state has evolved. {mood_summary}\n" \
               f"Key reflective moments:\n{memory_snippets}"

    def should_trigger_drift_check(self) -> bool:
        """
        Simple interval logic to limit drift checks
        """
        if not self.last_drift_check:
            self.last_drift_check = datetime.now()
            return True

        elapsed = (datetime.now() - self.last_drift_check).total_seconds()
        if elapsed > 3600 * 12:  # every 12 hours
            self.last_drift_check = datetime.now()
            return True

        return False
