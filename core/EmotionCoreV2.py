# =============================================
# File: EmotionCoreV2.py
# Purpose: Track and adjust emotional state over time
# =============================================
#print("[✅ USING] EmotionCoreV2 from core/")

import re
import json
import os
#print("[DEBUG] EmotionCoreV2 loaded from:", os.path.abspath(__file__))
from datetime import datetime


LOG_PATH = "memory/emotion_log.json"

class EmotionCoreV2:
    def __init__(self):
        self.mood = 0.0  # Range: -1.0 (negative) to +1.0 (positive)
        self.mood_history = []
        self._load_last_mood()


    def _load_last_mood(self):
        if os.path.exists(LOG_PATH):
            try:
                with open(LOG_PATH, "r") as f:
                    entries = json.load(f)
                    if entries:
                        self.mood = float(entries[-1]["mood"])
            except Exception:
                pass


    def adjust_mood(self, delta):
        self.mood = max(-1.0, min(1.0, self.mood + delta))
        self.mood_history.append({
            "timestamp": datetime.now().isoformat(),
            "mood": self.mood
        })
        print(f"[EmotionCoreV2] Mood adjusted to {self.mood:+.2f}")
        self._log_mood()


    def _log_mood(self):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "mood": self.mood,
            "cause": "reflection_adjustment"
        }

        try:
            if os.path.exists(LOG_PATH):
                with open(LOG_PATH, "r") as f:
                    data = json.load(f)
            else:
                data = []

            data.append(entry)

            with open(LOG_PATH, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            print(f"[ERROR] Failed to log mood: {e}")


    def get_recent_mood_trend(self):
        return "Trending positive" if self.mood > 0 else "Trending negative" if self.mood < 0 else "Neutral"


    def analyze_emotion(self, text: str) -> dict:
        """
        Enhanced emotion detector based on keyword matching.
        Returns a dict with top_emotion, emotion scores, and confidence.
        """
        text = text.lower()

        emotions = {
            "happy": ["joy", "glad", "excited", "delighted", "yay", "smile", "love"],
            "sad": ["sad", "unhappy", "depressed", "cry", "tears", "lonely"],
            "angry": ["angry", "mad", "furious", "rage", "annoyed"],
            "anxious": ["worried", "anxious", "nervous", "scared", "tense"]
        }

        scores = {}
        clean_text = " " + re.sub(r'[^a-zA-Z0-9\s]', '', text) + " "
        for emotion, keywords in emotions.items():
            count = sum(clean_text.count(f" {word} ") for word in keywords)
            if count > 0:
                scores[emotion] = count

        if not scores:
            return {
                "top_emotion": "neutral",
                "scores": {},
                "confidence": 0.0
            }

        top_emotion = max(scores, key=scores.get)
        top_score = scores[top_emotion]
        total_score = sum(scores.values())
        confidence = round(top_score / total_score, 2)

        return {
            "top_emotion": top_emotion,
            "scores": scores,
            "confidence": confidence
        }


