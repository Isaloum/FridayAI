# ==============================================
# File: core/MoodManagerCore.py
# Purpose: Tracks and persists emotional trends over time
# ==============================================

import os
import json
from datetime import datetime
from collections import defaultdict
from typing import Dict, Optional


class MoodManagerCore:
    """
    Tracks and persists emotional trends over time.
    Can be used by FridayAI to adjust tone, behavior, or interventions.
    """

    def __init__(self, state_file: str = "mood_state.json"):
        self.state_file = state_file
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_state(self):
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=4)
        except Exception as e:
            print(f"⚠️ Mood state save failed: {e}")

    def update_mood(self, session_id: str, emotion: str):
        now = datetime.now().isoformat()
        if session_id not in self.state:
            self.state[session_id] = []

        self.state[session_id].append({
            "timestamp": now,
            "emotion": emotion
        })
        self._save_state()

    def get_current_session_mood(self, session_id: str) -> Optional[str]:
        if session_id not in self.state or not self.state[session_id]:
            return None

        # Count frequency in current session
        mood_freq = defaultdict(int)
        for entry in self.state[session_id]:
            mood_freq[entry["emotion"]] += 1

        return max(mood_freq, key=mood_freq.get)

    def get_mood_report(self) -> Dict:
        trend = defaultdict(int)
        for session in self.state.values():
            for entry in session:
                trend[entry["emotion"]] += 1

        if not trend:
            return {"dominant_mood": "neutral", "distribution": {}}

        dominant = max(trend, key=trend.get)
        return {
            "dominant_mood": dominant,
            "distribution": dict(trend),
            "total_sessions": len(self.state)
        }

    def reset(self):
        self.state = {}
        self._save_state()
