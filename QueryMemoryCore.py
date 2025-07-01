
# QueryMemoryCore.py
# ------------------
# Enhanced memory querying module with tag fuzzing and emotion metadata support

from datetime import datetime, timedelta
from typing import List, Dict
import re

class QueryMemoryCore:
    def __init__(self, memory_core):
        self.memory = memory_core

    def fetch_recent(self, limit: int = 10, within_minutes: int = 1440) -> List[Dict]:
        cutoff = datetime.now() - timedelta(minutes=within_minutes)
        all_entries = self.memory.get_all()
        filtered = [entry for entry in all_entries if datetime.fromisoformat(entry['timestamp']) >= cutoff]
        sorted_entries = sorted(filtered, key=lambda e: e['timestamp'], reverse=True)
        return sorted_entries[:limit]

    def search_by_tag(self, tag: str, days: int = 30) -> List[Dict]:
        cutoff = datetime.now() - timedelta(days=days)
        entries = self.memory.get_all()
        tag = tag.lower()
        return [
            e for e in entries
            if any(tag in t.lower() for t in e.get('tags', []))
            and datetime.fromisoformat(e['timestamp']) > cutoff
        ]

    def keyword_search(self, keyword: str, days: int = 30) -> List[Dict]:
        cutoff = datetime.now() - timedelta(days=days)
        entries = self.memory.get_all()
        return [
            e for e in entries 
            if keyword.lower() in str(e.get('value', '')).lower() 
            and datetime.fromisoformat(e['timestamp']) > cutoff
        ]

    def summarize(self, tag: str = None, days: int = 30) -> str:
        if tag:
            relevant = self.search_by_tag(tag, days)
        else:
            relevant = self.fetch_recent(limit=50, within_minutes=days*1440)
        if not relevant:
            return "No relevant memories found."

        summary = f"In the last {days} days, here’s what I recall"
        summary += f" about '{tag}':\n" if tag else ":\n"

        for item in relevant[:10]:
            date = item['timestamp'].split("T")[0]
            summary += f"- [{date}] {str(item.get('value', ''))[:80]}...\n"

        return summary.strip()

    def reflect_emotions(self, days: int = 30) -> str:
        data = self.memory.get_all()
        cutoff = datetime.now() - timedelta(days=days)
        recent = [e for e in data if datetime.fromisoformat(e['timestamp']) > cutoff]

        emotional_trends = {}
        for entry in recent:
            meta_emotion = entry.get("metadata", {}).get("emotion", {})
            if isinstance(meta_emotion, dict):
                for k, v in meta_emotion.items():
                    emotional_trends[k] = emotional_trends.get(k, 0) + v
            for tag in entry.get("tags", []):
                if tag in ["happy", "sad", "angry", "anxious", "excited"]:
                    emotional_trends[tag] = emotional_trends.get(tag, 0) + 1

        if not emotional_trends:
            return "No emotional trends found."

        sorted_trend = sorted(emotional_trends.items(), key=lambda x: x[1], reverse=True)
        out = f"🧠 Emotional trends for the last {days} days:\n"
        for emotion, count in sorted_trend:
            out += f"• {emotion}: {count} entries\n"
        return out

    def get_frequent_topics(self, top_n: int = 5) -> List[str]:
        entries = self.memory.get_all()
        tag_freq = {}
        emotion_freq = {}

        for e in entries:
            for tag in e.get("tags", []):
                tag_freq[tag] = tag_freq.get(tag, 0) + 1
            for emo, v in e.get("metadata", {}).get("emotion", {}).items():
                emotion_freq[emo] = emotion_freq.get(emo, 0) + v

        combined = {**tag_freq, **emotion_freq}
        sorted_items = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return [tag for tag, _ in sorted_items[:top_n]]

    def query_memory(self, query: str, days: int = 30) -> List[Dict]:
        cutoff = datetime.now() - timedelta(days=days)
        entries = self.memory.get_all()
        keywords = [word.lower() for word in re.findall(r'\b\w{4,}\b', query)]

        results = []
        for e in entries:
            if datetime.fromisoformat(e['timestamp']) < cutoff:
                continue
            content = str(e.get("value", "")).lower()
            tags = [t.lower() for t in e.get("tags", [])]
            if any(kw in content or kw in tags for kw in keywords):
                results.append(e)

        sorted_results = sorted(results, key=lambda x: x['timestamp'], reverse=True)
        return sorted_results[:10]
