# ==============================================
# File: core/pregnancy/PregnancyMentalHealthAgent.py
# Purpose: Monitor emotional trends and offer mental health support
# ==============================================

from datetime import datetime, timedelta

class PregnancyMentalHealthAgent:
    def __init__(self, memory, profile):
        self.memory = memory
        self.profile = profile

    def get_support_plan(self):
        # Analyze recent emotional trends
        recent_emotions = self._get_recent_emotions()
        primary_emotion = self._most_common_emotion(recent_emotions)

        # Suggest support plan based on dominant emotion
        return self._suggest_plan(primary_emotion)

    def _get_recent_emotions(self, days=7):
        recent_entries = self.memory.query_memories(since_hours=days*24)
        return [entry.get("emotion", "neutral") for entry in recent_entries if entry.get("type") == "pregnancy_log"]

    def _most_common_emotion(self, emotions):
        if not emotions:
            return "neutral"
        return max(set(emotions), key=emotions.count)

    def _suggest_plan(self, emotion):
        plans = {
            "sad": "Try writing a gratitude note or calling a supportive friend.",
            "anxious": "Practice a 5-minute breathing exercise and take a short walk.",
            "angry": "Consider journaling or doing a physical release like stretching.",
            "neutral": "Take a moment to check in with yourself — maybe write a small reflection.",
            "happy": "Celebrate your joy — share it with your partner or log it in your journal."
        }
        return f"🧠 Detected Emotion: {emotion}\n🩺 Suggested Self-Care Steps:\n- {plans.get(emotion, 'Take a mindful pause and do something comforting.')}"
