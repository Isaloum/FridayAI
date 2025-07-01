# =====================================
# ReflectionCore.py – Pattern-Aware Memory Reflection
# =====================================

from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import List, Dict
import re

class ReflectionCore:
    """Analyzes memory + emotion patterns to surface deeper insights."""

    def __init__(self, memory_core):
        self.memory = memory_core

    def reflect_on_patterns(self, days: int = 7, threshold: int = 2) -> str:
        """
        Detects recurring emotion-topic combinations and surfaces meaningful reflections.
        """
        cutoff = datetime.now() - timedelta(days=days)
        recent_entries = [
            entry for entry in self.memory.get_all()
            if datetime.fromisoformat(entry.get("timestamp", "")) > cutoff
        ]

        topic_emotion_map = defaultdict(Counter)
        for entry in recent_entries:
            tags = entry.get("metadata", {}).get("tags", [])
            emotion = entry.get("metadata", {}).get("emotion", {})
            for tag in tags:
                for e_key, e_val in emotion.items():
                    topic_emotion_map[tag][e_key] += e_val

        reflections = []
        for tag, emotions in topic_emotion_map.items():
            for e_type, count in emotions.items():
                if count >= threshold:
                    reflections.append((tag, e_type, count))

        if not reflections:
            return "You've had a variety of experiences lately, but no strong patterns stood out. Keep checking in."

        # Format natural reflections
        response = "Here’s something I’ve noticed recently:\n"
        for tag, emotion, count in reflections[:5]:
            response += f"– You’ve brought up **{tag}** while feeling **{emotion}** {count} times.\n"
        response += "\nWant to unpack any of that?"

        return response.strip()
