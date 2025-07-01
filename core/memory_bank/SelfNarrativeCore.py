# =============================================
# File: SelfNarrativeCore.py
# Purpose: Maintain Friday's self-narrative and reflection log
# =============================================

import json
from pathlib import Path
from datetime import datetime

NARRATIVE_PATH = Path("core/memory_bank/self_narrative_log.json")

class SelfNarrativeCore:
    """
    Logs introspective events, reflections, or key life events into a growing narrative.
    """

    def __init__(self):
        self.path = NARRATIVE_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump([], f)

    def log_event(self, content: str, kind="reflection"):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "kind": kind,
            "content": content
        }
        with open(self.path, 'r+', encoding='utf-8') as f:
            data = json.load(f)
            data.append(entry)
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()

        print(f"📝 [Narrative] Logged new {kind} at {entry['timestamp']}")
