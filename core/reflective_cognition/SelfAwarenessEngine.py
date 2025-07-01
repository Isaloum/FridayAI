# =============================================
# File: SelfAwarenessEngine.py
# Purpose: Analyze emotional/memory trends and generate reflective insights
# =============================================

from datetime import datetime
from statistics import mean

class SelfAwarenessEngine:
    def __init__(self, emotion_core, memory_core):
        self.emotion_core = emotion_core
        self.memory_core = memory_core

    def generate_self_reflection(self) -> str:
        recent_memories = self.memory_core.query_recent(limit=10)
        if not recent_memories:
            return "I don't have enough behavioral context to reflect right now."

        emotions = [m.get("emotion", "neutral") for m in recent_memories]
        mood_values = [self._map_emotion_to_value(e) for e in emotions]

        avg_mood = mean(mood_values)
        tone = self._describe_mood(avg_mood)

        return f"As of {datetime.now().isoformat()}, my emotional tone is {tone}. Recent memories suggest my behavior is trending in a {tone} direction."

    def _map_emotion_to_value(self, emotion: str) -> float:
        mapping = {
            "joy": 1.0,
            "happy": 0.7,
            "neutral": 0.0,
            "sad": -0.5,
            "anger": -1.0,
            "fear": -0.8,
            "danger": -0.6,
            "urgent": -0.4,
        }
        return mapping.get(emotion.lower(), 0.0)

    def _describe_mood(self, avg: float) -> str:
        if avg >= 0.6:
            return "positive"
        elif avg >= 0.1:
            return "slightly positive"
        elif avg > -0.1:
            return "neutral"
        elif avg > -0.6:
            return "slightly negative"
        else:
            return "negative"

    def should_trigger_drift_check(self) -> bool:
        # Placeholder logic – in future, trigger based on volatility or critical reflection tags
        return True
