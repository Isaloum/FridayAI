# BeliefExplanationCore.py – Justification Engine for Trait Evolution

import os
import json
from datetime import datetime, timedelta

class BeliefExplanationCore:
    def __init__(self, log_dir="./memory/belief_updates"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def log_belief_change(self, user_id: str, trait: str, direction: str, reason: str):
        log_file = os.path.join(self.log_dir, f"{user_id}_changes.json")
        entry = {
            "timestamp": datetime.now().isoformat(),
            "trait": trait,
            "direction": direction,
            "reason": reason
        }

        try:
            with open(log_file, "r") as f:
                data = json.load(f)
        except:
            data = []

        data.append(entry)

        with open(log_file, "w") as f:
            json.dump(data, f, indent=4)

    def explain_last_change(self, user_id: str, days=30) -> str:
        log_file = os.path.join(self.log_dir, f"{user_id}_changes.json")
        if not os.path.exists(log_file):
            return "I haven't updated any of my beliefs yet."

        try:
            with open(log_file, "r") as f:
                data = json.load(f)
        except:
            return "I can't retrieve my belief history right now."

        recent = []
        cutoff = datetime.now() - timedelta(days=days)
        for entry in reversed(data):
            ts = datetime.fromisoformat(entry["timestamp"])
            if ts >= cutoff:
                recent.append(entry)
            if len(recent) >= 3:
                break

        if not recent:
            return "No recent updates to my beliefs within the past month."

        summary = []
        for e in recent:
            summary.append(f"I became {e['direction']} {e['trait']} because you said: \"{e['reason']}\"")

        return "Here's how I've changed: " + " ".join(summary)
