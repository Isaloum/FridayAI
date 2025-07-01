# ========================================
# File: core/MoodPredictor.py
# Purpose: Forecast emotional trend based on past moods
# ========================================

from collections import Counter
from datetime import datetime
import math

class MoodPredictor:
    def __init__(self, mood_core):
        self.mood_core = mood_core

    def predict_trend(self, days=7) -> str:
        history = self.mood_core.get_mood_history(days=days)
        if not history:
            return "Mood history unavailable."

        moods = [entry["mood"] for entry in history if "mood" in entry]
        if not moods:
            return "No moods found in history."

        counts = Counter(moods)
        most_common, freq = counts.most_common(1)[0]
        trend = self._detect_change(history)

        return f"📈 Trending toward **{most_common}** ({trend})"

    def _detect_change(self, history: list) -> str:
        timeline = sorted(history, key=lambda x: x["timestamp"])
        recent = timeline[-3:]
        mood_ids = [x["mood"] for x in recent]

        if len(set(mood_ids)) == 1:
            return "stable"
        elif mood_ids[-1] == mood_ids[-2] and mood_ids[-2] != mood_ids[0]:
            return "swing"
        else:
            return "volatile"
