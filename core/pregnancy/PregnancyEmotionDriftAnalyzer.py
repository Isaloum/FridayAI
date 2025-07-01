# ==============================================
# File: core/pregnancy/PregnancyEmotionDriftAnalyzer.py
# Purpose: Detect changes in emotional patterns over time
# ==============================================

from datetime import datetime, timedelta
from collections import Counter

class PregnancyEmotionDriftAnalyzer:
    def __init__(self, memory):
        self.memory = memory

    def analyze_drift(self, days=14):
        entries = self.memory.query_memories(filter_tags=["emotion_trace"], since_hours=days*24)
        if not entries:
            return "No emotional data available."

        timeline = {}
        for e in entries:
            day = e["timestamp"][:10]
            timeline.setdefault(day, []).append(e["emotion"])

        summaries = []
        for day, emotions in sorted(timeline.items()):
            counts = Counter(emotions)
            top = counts.most_common(1)[0]
            summaries.append(f"{day}: {top[0]} ({top[1]} entries)")

        return "📈 Emotion Drift (last 2 weeks):\n" + "\n".join(summaries)
