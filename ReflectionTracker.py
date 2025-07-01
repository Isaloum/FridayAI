# ReflectionTracker.py – Tone Drift Analysis for FridayAI

import os
import json
from datetime import datetime, timedelta

class ReflectionTracker:
    def __init__(self, profile_path="./memory/personality_profile.json", log_dir="./memory/reflections"):
        self.profile_path = profile_path
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def load_profile_history(self, days=30):
        """Load old profiles from reflection logs"""
        snapshots = []
        cutoff = datetime.now() - timedelta(days=days)

        for file in os.listdir(self.log_dir):
            try:
                timestamp = datetime.strptime(file.replace(".json", ""), "%Y-%m-%d")
                if timestamp >= cutoff:
                    with open(os.path.join(self.log_dir, file), 'r') as f:
                        profile = json.load(f)
                        snapshots.append((timestamp, profile))
            except:
                continue

        return sorted(snapshots, key=lambda x: x[0])

    def save_today_profile(self, profile: dict):
        today = datetime.now().strftime("%Y-%m-%d")
        path = os.path.join(self.log_dir, f"{today}.json")
        with open(path, 'w') as f:
            json.dump(profile, f, indent=4)

    def reflect(self, current_profile: dict) -> str:
        """Analyze drift and return reflection text"""
        history = self.load_profile_history()
        if not history:
            self.save_today_profile(current_profile)
            return "This is my first personality snapshot. I’ll start observing myself from here."

        past_profile = history[0][1]  # Oldest within range

        diffs = {}
        for trait in current_profile:
            change = round(current_profile[trait] - past_profile[trait], 2)
            if abs(change) >= 0.05:
                direction = "more" if change > 0 else "less"
                diffs[trait] = (direction, abs(change))

        self.save_today_profile(current_profile)

        if not diffs:
            return "Over the past month, my personality has remained fairly stable."

        # Human-friendly translation
        trait_map = {
            "warmth": "compassionate",
            "humor": "witty",
            "precision": "analytical",
            "formality": "formal"
        }

        parts = [
            f"{direction} {trait_map.get(trait, trait)} ({delta})"
            for trait, (direction, delta) in diffs.items()
        ]
        summary = ", ".join(parts)
        return f"Over the past month, I’ve become {summary}."

