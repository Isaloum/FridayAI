# EmotionalJournal.py
# --------------------------------------------------
# Tracks and summarizes emotional states over time

import json
import os
from datetime import datetime, timedelta
from collections import defaultdict, Counter

class EmotionalJournal:
    def __init__(self, memory_path='long_term_memory.json', journal_path='emotional_journal.json'):
        self.memory_path = memory_path
        self.journal_path = journal_path
        self.journal = self._load_journal()

    def _load_journal(self):
        if not os.path.exists(self.journal_path):
            return {}
        with open(self.journal_path, 'r') as f:
            return json.load(f)

    def _save_journal(self):
        with open(self.journal_path, 'w') as f:
            json.dump(self.journal, f, indent=2)

    def analyze_day(self, date_str=None):
        if not os.path.exists(self.memory_path):
            return

        with open(self.memory_path, 'r') as f:
            entries = json.load(f)

        if not date_str:
            date_str = datetime.now().strftime('%Y-%m-%d')

        day_emotions = Counter()
        found = 0

        for entry in entries:
            timestamp = entry.get("timestamp", "")
            if not timestamp.startswith(date_str):
                continue

            reply = entry.get("reply", "").lower()
            for mood in ["happy", "sad", "angry", "love", "confused", "excited", "lonely"]:
                if mood in reply:
                    day_emotions[mood] += 1
                    found += 1

        if found:
            summary = {
                "date": date_str,
                "total_mentions": found,
                "emotions": dict(day_emotions),
                "dominant": day_emotions.most_common(1)[0][0] if day_emotions else "neutral"
            }
            self.journal[date_str] = summary
            self._save_journal()
            return summary
        return None

    def summarize_range(self, days_back=7):
        today = datetime.now()
        result = []
        for i in range(days_back):
            date_str = (today - timedelta(days=i)).strftime('%Y-%m-%d')
            if date_str in self.journal:
                result.append(self.journal[date_str])
        return result[::-1]  # chronological

# For manual test
if __name__ == "__main__":
    ej = EmotionalJournal()
    today_summary = ej.analyze_day()
    if today_summary:
        print("📝 Today's Emotion Summary:", today_summary)
    else:
        print("No emotional entries found today.")

    print("\n📅 Past 7-Day Journal:")
    for entry in ej.summarize_range():
        print(f"{entry['date']}: {entry['dominant']} ({entry['emotions']})")
