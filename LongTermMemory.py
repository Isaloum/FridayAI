# LongTermMemory.py
# ----------------------------------
# Logs and retrieves conversation history for FridayAI

import json
import os
from datetime import datetime, timedelta
from typing import List, Dict

class LongTermMemory:
    def __init__(self, filepath: str = "long_term_memory.json"):
        self.filepath = filepath
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w') as f:
                json.dump([], f)

    def fetch_recent(self, limit=5, within_minutes=60):
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                memory = json.load(f)
        except:
            return []

        cutoff = datetime.now() - timedelta(minutes=within_minutes)
        recent = [
            m for m in memory
            if "timestamp" in m and datetime.fromisoformat(m["timestamp"]) > cutoff
        ]
        return recent[-limit:]


    def store(self, user_input: str, reply: str, context: str = "") -> None:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "reply": reply,
            "context": context
        }
        data = self._load()
        data.append(entry)
        self._save(data)

    def search(self, keyword: str) -> List[Dict]:
        data = self._load()
        return [entry for entry in data if keyword.lower() in entry["user_input"].lower() or keyword.lower() in entry["reply"].lower()]

    def summarize_recent(self, limit: int = 5) -> str:
        data = self._load()[-limit:]
        summary = "\n".join([f"{entry['timestamp'][:19]} — You: {entry['user_input']} | Friday: {entry['reply']}" for entry in data])
        return summary if summary else "No memory entries yet."

    def _load(self) -> List[Dict]:
        with open(self.filepath, 'r') as f:
            return json.load(f)

    def _save(self, data: List[Dict]) -> None:
        with open(self.filepath, 'w') as f:
            json.dump(data, f, indent=2)
