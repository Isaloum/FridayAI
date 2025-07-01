# =============================================
# File: SelfNarrativeCore.py
# Purpose: Store, retrieve, and narrate Friday's own day, memory, and state
# =============================================

from datetime import datetime
import json
import os

SELF_STATE_PATH = "memory_store/self_story/friday_narrative.json"

class SelfNarrativeCore:
    def __init__(self):
        self.state = {
            "date": str(datetime.now().date()),
            "events": [],
            "reflections": [],
            "tool_uses": [],
            "mood": "neutral"
        }
        self._load()

    def _load(self):
        if os.path.exists(SELF_STATE_PATH):
            with open(SELF_STATE_PATH, "r") as f:
                self.state = json.load(f)

    def _save(self):
        os.makedirs(os.path.dirname(SELF_STATE_PATH), exist_ok=True)
        with open(SELF_STATE_PATH, "w") as f:
            json.dump(self.state, f, indent=2)

    def log_event(self, event: str, kind: str = "event", source: str = None):
        entry = {
            "time": datetime.now().isoformat(),
            "type": kind,
            "text": event
        }
        if source:
            entry["source"] = source

        if kind == "tool":
            self.state["tool_uses"].append(entry)
        elif kind == "reflection":
            self.state["reflections"].append(entry)
        else:
            self.state["events"].append(entry)

        self._save()


    def update_mood(self, mood: str):
        self.state["mood"] = mood
        self._save()

    def summarize_day(self) -> str:
        return (
            f"🗓️ Friday's Self-Narrative ({self.state['date']}):\n"
            f"- Mood: {self.state['mood']}\n"
            f"- Events: {len(self.state['events'])} logged\n"
            f"- Reflections: {len(self.state['reflections'])} noted\n"
            f"- Tools used: {len(self.state['tool_uses'])}\n"
        )

    def get_raw_state(self):
        return self.state
