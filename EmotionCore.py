# =====================================
# EmotionCore.py - Emotion Detection & Trend Tracker
# =====================================
import os
import json
from datetime import datetime, timedelta
from collections import Counter

class EmotionCore:
    """Tracks emotional patterns, logs moods, and provides summaries."""

    def __init__(self, emotion_log_file='emotion_log.json'):
        self.emotion_log_file = emotion_log_file
        self.entries = self._load_log()
        
    def analyze(self, text):
        """Very basic emotion keyword matcher."""
        keywords = {
            "happy": ["glad", "joy", "love", "excited", "grateful"],
            "sad": ["depressed", "down", "cry", "lonely", "hopeless"],
            "angry": ["mad", "furious", "irritated", "pissed", "rage"],
            "anxious": ["nervous", "worried", "tense", "afraid", "overwhelmed"],
            "excited": ["thrilled", "pumped", "ecstatic", "elated"]
        }

        text = text.lower()
        scores = {}
        for emotion, words in keywords.items():
            scores[emotion] = sum(1 for w in words if w in text)

        # Optional: log result
        if any(scores.values()):
            self.log_emotion({k: v for k, v in scores.items() if v > 0})

        return {k: v for k, v in scores.items() if v > 0}


    def _load_log(self):
        """Load existing emotion log or initialize empty."""
        if os.path.exists(self.emotion_log_file):
            with open(self.emotion_log_file, 'r') as f:
                return json.load(f)
        return []

    def _save_log(self):
        """Save emotion log to disk."""
        with open(self.emotion_log_file, 'w') as f:
            json.dump(self.entries, f, indent=2)

    def log_emotion(self, emotion_dict):
        """
        Log detected emotions with a timestamp.
        :param emotion_dict: Example: {'happy': 1, 'relieved': 1}
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'emotions': emotion_dict
        }
        self.entries.append(entry)
        self._save_log()

    def get_latest_emotions(self):
        """Return the most recent logged emotional state."""
        if not self.entries:
            return {}
        return self.entries[-1]["emotions"]

    def get_emotion_trend(self, days=7):
        """
        Summarize emotional trends over the last N days.
        :param days: Number of days to look back
        :return: {'date': ..., 'total_mentions': int, 'emotions': {..}, 'dominant': str}
        """
        cutoff = datetime.now() - timedelta(days=days)
        trend_entries = [
            e for e in self.entries
            if datetime.fromisoformat(e["timestamp"]) >= cutoff
        ]

        all_emotions = Counter()
        for entry in trend_entries:
            all_emotions.update(entry["emotions"])

        dominant = all_emotions.most_common(1)[0][0] if all_emotions else "none"

        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "total_mentions": sum(all_emotions.values()),
            "emotions": dict(all_emotions),
            "dominant": dominant
        }

    def clear_log(self):
        """Reset emotion log (for testing or wipe)."""
        self.entries = []
        self._save_log()
        
    def analyze_emotion(self, text):
        return "neutral"


# ======================
# Example Usage (if run directly)
# ======================
if __name__ == "__main__":
    ec = EmotionCore()
    ec.log_emotion({"happy": 1, "hopeful": 1})
    print("Latest:", ec.get_latest_emotions())
    print("Trend:", ec.get_emotion_trend(days=3))
