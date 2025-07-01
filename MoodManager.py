# =====================================
# MoodManager.py
# Tracks long-term emotional state of the user.
# =====================================

from collections import deque, Counter
from typing import Optional, List
import datetime

class MoodManager:
    def __init__(self, max_history: int = 20):
        """
        Keeps a rolling history of detected emotions.
        Adjusts mood based on frequency and recency.
        """
        self.history = deque(maxlen=max_history)
        self.timestamped = []  # For timeline tracking

    def update_mood(self, emotions: dict):
        """
        Store new emotion reading.
        Example input: {'sadness': 0.9, 'anger': 0.2}
        """
        if not emotions:
            return

        top_emotion = max(emotions.items(), key=lambda x: x[1])[0]
        self.history.append(top_emotion)
        self.timestamped.append((datetime.datetime.now(), top_emotion))

    def get_current_mood(self) -> Optional[str]:
        """
        Return the dominant mood based on history.
        """
        if not self.history:
            return None

        freq = Counter(self.history)
        most_common = freq.most_common(1)
        return most_common[0][0] if most_common else None

    def get_mood_report(self) -> dict:
        """
        Returns a full mood diagnostic.
        """
        mood_counts = Counter(self.history)
        timeline = [(ts.strftime("%Y-%m-%d %H:%M"), mood) for ts, mood in self.timestamped]

        return {
            "mood_trend": mood_counts.most_common(),
            "recent_mood": list(self.history),
            "timeline": timeline
        }

# Test it
if __name__ == "__main__":
    mm = MoodManager()
    mm.update_mood({'sadness': 0.9})
    mm.update_mood({'joy': 0.6})
    mm.update_mood({'joy': 0.9})
    mm.update_mood({'anger': 0.7})

    print("Current Mood:", mm.get_current_mood())
    print("\nFull Mood Report:")
    print(mm.get_mood_report())
