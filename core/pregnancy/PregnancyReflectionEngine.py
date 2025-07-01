# ==============================================
# File: core/pregnancy/PregnancyReflectionEngine.py
# Purpose: Summarizes emotional journey with disclaimers and confidence
# ==============================================

from datetime import datetime, timedelta
from core.pregnancy.PregnancyConfidenceScorer import ConfidenceScorer

class PregnancyReflectionEngine:
    """
    Reflects on the emotional log entries related to pregnancy for the past week.
    Adds emotional confidence and sensitive disclaimers for trust and safety.
    """
    def __init__(self, memory):
        self.memory = memory
        self.confidence_scorer = ConfidenceScorer()

    def summarize_week(self):
        # Pull last week's pregnancy-specific memories
        entries = self.memory.query_memories(filter_tags=["pregnancy_log"], since_hours=168)
        
        anchor_events = self.memory.query_memories(filter_tags=["anchor_event"], since_hours=168)
        

        if anchor_events:
            lines.append("\n🔖 Major Emotional Events (Anchors):")
            for anchor in anchor_events:
                lines.append(f"- [{anchor.get('emotion')}] \"{anchor.get('text')}\" during {anchor.get('trimester')} trimester.")

        if not entries:
            return "No recent pregnancy reflections logged."

        mood_summary = {}
        lines = []

        for entry in entries:
            text = entry.get("text", "")
            mood = entry.get("emotion", "neutral")
            tri = entry.get("trimester", "None")
            confidence = self.confidence_scorer.score(text)
            label = self.confidence_scorer.label(confidence)

            lines.append(f"- You mentioned feeling '{text}' ({mood}) during the {tri} trimester. Confidence: {label}.")
            mood_summary[mood] = mood_summary.get(mood, 0) + 1

        most_common = max(mood_summary, key=mood_summary.get, default="neutral")
        total = sum(mood_summary.values())
        percent = int((mood_summary.get(most_common, 0) / total) * 100)

        reflection = [
            "Here’s what your past week looked like emotionally:\n",
            *lines,
            f"\n🌈 This week you mostly experienced: {most_common} ({percent}% of entries)",
            "💬 I'm here for all of it. Would you like to reflect on why this might be?",
            "",
            "🤖 *These reflections are AI-generated to support—not replace—medical advice.*",
            "📊 *Confidence indicators help you understand how sure I am. When unsure, reach out to a trusted human.*"
        ]

        return "\n".join(reflection)
