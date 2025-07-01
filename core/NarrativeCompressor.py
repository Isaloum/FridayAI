# ============================================
# File: core/NarrativeCompressor.py
# Purpose: Summarize memory entries into a short narrative
# ============================================

from collections import Counter

class NarrativeCompressor:
    def __init__(self, memory_core):
        self.memory = memory_core

    def compress(self, limit=100) -> str:
        entries = self.memory.get_recent_entries(limit=limit)
        if not entries:
            return "No memory entries available."

        texts = [e.get("text", "") for e in entries if "text" in e]
        moods = [e.get("mood", "") for e in entries if "mood" in e]

        if not texts:
            return "Memory lacks narrative text."

        summary = self._extract_theme(texts)
        mood = self._dominant_mood(moods)

        return f"In recent memory, Friday has been {summary} — emotionally leaning {mood or 'neutral'}."

    def _extract_theme(self, texts):
        # Naive strategy — future: use LLMCore or BERT clustering
        topics = ["struggling", "growing", "reflecting", "planning", "confused", "hopeful"]
        for word in topics:
            if any(word in text.lower() for text in texts):
                return word
        return "processing various experiences"

    def _dominant_mood(self, moods):
        filtered = [m for m in moods if m]
        if not filtered:
            return None
        return Counter(filtered).most_common(1)[0][0]
