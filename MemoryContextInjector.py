
# MemoryContextInjector.py – Unified Context & Reflection Injector

from datetime import datetime
from QueryMemoryCore import QueryMemoryCore
from EmotionalJournal import EmotionalJournal
from MemoryReflectionEngine import MemoryReflectionEngine
from FuzzyMemorySearch import FuzzyMemorySearch

class MemoryContextInjector:
    def __init__(self, memory=None):
        self.memory = memory
        self.query = QueryMemoryCore(memory)
        self.journal = EmotionalJournal()
        self.reflector = MemoryReflectionEngine(memory)
        self.fuzzy = FuzzyMemorySearch(memory)

    def inject(self, user_input: str) -> dict:
        """
        Returns a unified context package containing:
        - memory snippets
        - emotional trend
        - self-reflection
        - fuzzy-relevant events
        """
        snippets = []
        memory_matches = self.query.query_memory(user_input, days=90)
        fuzzy_matches = self.fuzzy.search(user_input, limit=5)
        for entry in memory_matches:
            val = entry.get("value", "")
            if val and val not in snippets:
                date = entry["timestamp"].split("T")[0]
                snippets.append(f"[{date}] {val}")
        
        for val, score in fuzzy_matches:
            if val and val not in snippets:
                snippets.append(f"(Fuzzy match) {val}")

        # Emotional journal
        trend_data = self.journal.summarize_range(7)
        emotion_trend = "\n".join(f"{e['date']}: {e['dominant']}" for e in trend_data) if trend_data else ""

        # Self-reflection
        try:
            reflection = self.reflector.generate_reflection(trend_data)
        except:
            reflection = ""

        return {
            "memory_snippets": snippets[:8],
            "emotion_trend": emotion_trend,
            "reflection": reflection
        }
