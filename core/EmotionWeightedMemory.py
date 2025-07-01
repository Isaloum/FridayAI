# =========================================
# File: EmotionWeightedMemory.py
# Purpose: Rank memories based on emotion + recency
# Dependencies: datetime
# =========================================

from datetime import datetime
import math

class EmotionWeightedMemory:
    def __init__(self, memory_core, emotion_core):
        self.memory = memory_core
        self.emotion = emotion_core

    def rank_memories(self, current_mood: str, top_n: int = 5) -> list:
        memories = self.memory.get_recent_entries(limit=100)
        ranked = []

        for entry in memories:
            text = entry.get("text", "")
            timestamp = entry.get("timestamp")
            if not text or not timestamp:
                continue

            # Score emotion similarity
            mood_score = self.emotion.compare_emotion(text, current_mood)  # float 0–1

            # Score recency
            time_score = self._time_decay(timestamp)

            # Weighted score: mood 70%, time 30%
            score = 0.7 * mood_score + 0.3 * time_score
            ranked.append((score, entry))

        # Sort by score descending
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in ranked[:top_n]]

    def _time_decay(self, timestamp: str) -> float:
        try:
            t = datetime.fromisoformat(timestamp)
            delta = (datetime.now() - t).total_seconds() / 3600  # hours ago
            return math.exp(-delta / 72)  # 72h half-life
        except Exception:
            return 0.0
