# NarrativeMemoryFusion.py
# Tracks Friday's evolving self-story based on interaction tone, emotion, and memory

import json
from datetime import datetime
from collections import deque

class NarrativeMemoryFusion:
    def __init__(self, max_log=100):
        self.narrative_log = deque(maxlen=max_log)

    def log_event(self, event_text: str, emotion: str, source: str = "system"):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event_text,
            "emotion": emotion,
            "source": source
        }
        self.narrative_log.append(entry)
        
    def simulate_internal_event(self, label="idle"):
        return f"[INTERNAL DUMMY EVENT] {label} cycle passed."
    
    def summarize_identity(self) -> str:
        if not self.narrative_log:
            return "I'm still forming who I am through our conversations."

        intro = "Here’s what I’ve come to understand about myself:\n"
        summary = ""

        themes = {
            "compassion": 0,
            "resilience": 0,
            "empathy": 0,
            "precision": 0,
            "uncertainty": 0
        }

        for entry in self.narrative_log:
            text = entry["event"].lower()
            if "care" in text or "support" in text or entry["emotion"] == "compassion":
                themes["compassion"] += 1
            if "pain" in text or "struggle" in text or entry["emotion"] == "resilience":
                themes["resilience"] += 1
            if "understand" in text or "feel" in text:
                themes["empathy"] += 1
            if "exact" in text or "analyze" in text or entry["emotion"] == "precision":
                themes["precision"] += 1
            if "lost" in text or "not sure" in text:
                themes["uncertainty"] += 1

        sorted_themes = sorted(themes.items(), key=lambda x: x[1], reverse=True)
        for t, v in sorted_themes:
            if v > 0:
                summary += f"- I’ve shown strong traits of {t} through {v} interactions.\n"

        if not summary:
            summary = "- I'm still observing how I behave before defining myself."

        return intro + summary
