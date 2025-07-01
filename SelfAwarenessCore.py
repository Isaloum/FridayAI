# SelfAwarenessCore.py
# Tracks Friday's own behavior over time and reflects on her tone evolution.

import json
from datetime import datetime, timedelta
from collections import deque

class SelfAwarenessCore:
    def __init__(self, memory_limit=50):
        self.behavior_log = deque(maxlen=memory_limit)

    def log_response_traits(self, tone_profile: dict, timestamp: str = None):
        entry = {
            "timestamp": timestamp or datetime.now().isoformat(),
            "warmth": tone_profile.get("warmth", 0.5),
            "humor": tone_profile.get("humor", 0.5),
            "formality": tone_profile.get("formality", 0.5),
            "precision": tone_profile.get("precision", 0.5)
        }
        self.behavior_log.append(entry)

    def analyze_recent_behavior(self, days: int = 7):
        cutoff = datetime.now() - timedelta(days=days)
        filtered = [
            e for e in self.behavior_log
            if datetime.fromisoformat(e["timestamp"]) > cutoff
        ]
        if not filtered:
            return {}

        total = len(filtered)
        sums = {"warmth": 0, "humor": 0, "formality": 0, "precision": 0}
        for e in filtered:
            for k in sums:
                sums[k] += e.get(k, 0)

        averages = {k: round(v / total, 3) for k, v in sums.items()}
        return averages

    def generate_self_reflection(self):
        stats = self.analyze_recent_behavior()
        if not stats:
            return "I don't have enough of my own behavior to reflect on yet."

        reflection = "I've been observing myself lately. Here's what I noticed:\n"
        for k, v in stats.items():
            if k == "warmth":
                if v < 0.4:
                    reflection += "- I’ve been a bit too cold lately. I’ll try to be warmer.\n"
                elif v > 0.75:
                    reflection += "- I’ve been very warm. I hope that’s felt comforting.\n"
            elif k == "formality":
                if v > 0.8:
                    reflection += "- I’ve been very formal. Maybe I should relax more.\n"
                elif v < 0.3:
                    reflection += "- I've been too casual. I might need to balance that.\n"
            elif k == "humor":
                if v > 0.7:
                    reflection += "- I’ve been playful a lot. Let me know if it’s ever too much.\n"
            elif k == "precision":
                if v > 0.8:
                    reflection += "- I’ve been extremely focused on details. Maybe too much.\n"
        return reflection.strip()

    def should_trigger_drift_check(self) -> bool:
        """
        Determines whether a belief drift check should be run.
        Example: every 5th session.
        """
        return hasattr(self, "session_count") and self.session_count % 5 == 0
