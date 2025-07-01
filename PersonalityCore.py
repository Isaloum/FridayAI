# PersonalityCore.py – Adaptive Trait Engine for FridayAI (Persistent Upgrade)

import json
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
from collections import Counter
from datetime import datetime

class PersonalityCore:
    def __init__(self, profile_path="./memory/personality_profile.json", debug=False):       
        self.profile_path = profile_path
        self.debug = debug
        self.profile = {
            "warmth": 0.5,
            "humor": 0.5,
            "precision": 0.5,
            "formality": 0.5
        }
        self.trend_log = []
        self.load_profile()

    def update_traits(self, emotion_list):
        counts = Counter(
            e["dominant"]
            for e in emotion_list
            if isinstance(e, dict) and "dominant" in e
        )

        total = sum(counts.values())

        if not total:
            return self.profile

        mood_weights = {
            "happy": {"warmth": 0.1, "humor": 0.1},
            "sad": {"warmth": 0.15, "formality": 0.05},
            "angry": {"precision": 0.1, "formality": 0.1},
            "anxious": {"warmth": 0.05, "precision": 0.1},
            "excited": {"humor": 0.15, "warmth": 0.1}
        }

        for mood, weight in mood_weights.items():
            freq = counts.get(mood, 0) / total
            for trait, delta in weight.items():
                self.profile[trait] += freq * delta

        for k in self.profile:
            self.profile[k] = max(0, min(1, round(self.profile[k], 2)))

        snapshot = {"timestamp": datetime.now().isoformat(), "profile": dict(self.profile)}
        self.trend_log.append(snapshot)
        self.save_profile()
        return self.profile

    def get_tone_description(self):
        tone = []
        if self.profile["warmth"] > 0.7:
            tone.append("compassionate")
        elif self.profile["warmth"] < 0.3:
            tone.append("detached")

        if self.profile["humor"] > 0.7:
            tone.append("witty")
        elif self.profile["humor"] < 0.3:
            tone.append("serious")

        if self.profile["precision"] > 0.7:
            tone.append("highly analytical")
        elif self.profile["precision"] < 0.3:
            tone.append("free-flowing")

        if self.profile["formality"] > 0.6:
            tone.append("formal")
        elif self.profile["formality"] < 0.3:
            tone.append("relaxed")

        return ", ".join(tone) or "balanced"

    def save_profile(self):
        try:
            os.makedirs(os.path.dirname(self.profile_path), exist_ok=True)
            with open(self.profile_path, 'w') as f:
                json.dump(self.profile, f, indent=4)
            if self.debug:
                print(f"[PersonalityCore] 💾 Saved profile to {self.profile_path}")
        except Exception as e:
            if self.debug:
                print(f"[PersonalityCore] ⚠️ Error saving profile: {e}")

    def load_profile(self):
        if os.path.exists(self.profile_path):
            try:
                with open(self.profile_path, 'r') as f:
                    self.profile = json.load(f)
                #print(f"[PersonalityCore] ✅ Loaded profile from {self.profile_path}")
            except Exception as e:
                print(f"[PersonalityCore] ⚠️ Failed to load profile: {e}")
        else:
            print(f"[PersonalityCore] ⚠️ No profile found, using default.")

    def get_profile(self):
        return self.profile
