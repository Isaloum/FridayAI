# ============================================
# File: DailySelfSummaryEngine.py
# Purpose: Generate daily emotional/memory summary for identity injection
# ============================================

from datetime import datetime
from MemoryScaffold import search_memory
from EmotionCore import EmotionCore
from SelfNarrativeCore import SelfNarrativeCore

class DailySelfSummaryEngine:
    @staticmethod
    def generate():
        # Load yesterday’s emotional + memory events
        now = datetime.now()
        yesterday = now.replace(hour=0, minute=0, second=0, microsecond=0)
        memory = search_memory("user", after=yesterday.isoformat())
        
        mood_stats = EmotionCore().summarize_emotions(memory)
        identity_lines = SelfNarrativeCore().summarize_events(memory)

        summary = f"""[Friday Daily Identity Snapshot — {now.strftime("%Y-%m-%d")}]
Mood: {mood_stats.get('dominant_mood', 'neutral')}
Emotional Range: {mood_stats.get('range', [])}
Memory Summary: {identity_lines[:3]}
"""

        return summary.strip()
