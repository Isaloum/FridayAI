# =====================================
# PlanningCore.py – Action-Oriented Pattern Planner
# =====================================

from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import Dict, List
import re

class PlanningCore:
    """Analyzes patterns and recommends goals or plans based on recurring problems."""

    def __init__(self, memory_core):
        self.memory = memory_core

    def suggest_plan(self, days: int = 7, threshold: int = 2) -> str:
        """
        Suggests actions based on recurring emotion-topic patterns.
        """
        cutoff = datetime.now() - timedelta(days=days)
        entries = [
            e for e in self.memory.get_all()
            if datetime.fromisoformat(e.get("timestamp", "")) > cutoff
        ]

        topic_emotion_map = defaultdict(Counter)
        for e in entries:
            tags = e.get("metadata", {}).get("tags", [])
            emotions = e.get("metadata", {}).get("emotion", {})
            for tag in tags:
                for em, val in emotions.items():
                    topic_emotion_map[tag][em] += val

        recommendations = []

        for tag, emotion_counter in topic_emotion_map.items():
            for emotion, count in emotion_counter.items():
                if count >= threshold:
                    if tag in ["pain", "injury", "sleep"]:
                        recommendations.append(
                            f"Would you like to plan a better rest or recovery routine for your {tag}?"
                        )
                    elif tag in ["work", "project", "focus"]:
                        recommendations.append(
                            f"Want help managing your schedule or reducing pressure from {tag}?"
                        )
                    elif tag in ["mood", "anxiety", "depression"]:
                        recommendations.append(
                            f"Should we build a mood journal or check-in routine to support you emotionally?"
                        )

        if not recommendations:
            return "No immediate changes needed. Things seem fairly balanced right now."

        # Prioritize no more than 3 suggestions
        return "Based on what I’ve seen, here’s how I could support you:\n\n" + "\n".join(f"– {r}" for r in recommendations[:3])
