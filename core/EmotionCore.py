import logging
from typing import Dict, List, Tuple
from collections import deque
import re

class EmotionCore:
    """
    Advanced emotional analysis engine with intensity detection,
    multi-emotion tagging, and persistent emotional state tracking.
    """

    def __init__(self, mood_window: int = 10):
        self.logger = logging.getLogger("FridayAI.EmotionCore")
        self.emotion_matrix = {
            "happy": ["happy", "joy", "excited", "great", "awesome", "fantastic", "amazing"],
            "sad": ["sad", "depressed", "unhappy", "crying", "miserable", "lonely"],
            "angry": ["angry", "mad", "furious", "annoyed", "pissed", "rage"],
            "stressed": ["stressed", "overwhelmed", "anxious", "nervous", "tense", "panic"],
            "love": ["love", "adore", "care", "affection", "cherish", "devotion"],
            "curious": ["curious", "wonder", "why", "how", "explain", "question"],
            "neutral": []
        }
        self.mood_history = deque(maxlen=mood_window)

    def analyze(self, text: str) -> str:
        """Quick analysis for single dominant emotion."""
        results = self.analyze_detailed(text)
        return results['top_emotion']

    def analyze_detailed(self, text: str) -> Dict:
        """
        Full emotional breakdown.
        :return: Dict with top_emotion, scores, and confidence
        """
        try:
            if not isinstance(text, str) or not text.strip():
                return {'top_emotion': 'neutral', 'scores': {}, 'confidence': 0.0}

            clean_text = " " + re.sub(r'[^a-zA-Z0-9\s]', '', text.lower()) + " "
            emotion_scores = {}

            for emotion, keywords in self.emotion_matrix.items():
                matches = sum(clean_text.count(f" {kw} ") for kw in keywords)
                emotion_scores[emotion] = matches

            # Remove zero-score entries
            filtered = {k: v for k, v in emotion_scores.items() if v > 0}
            top_emotion = max(filtered, key=filtered.get) if filtered else "neutral"
            top_score = filtered.get(top_emotion, 0)
            total_score = sum(filtered.values()) or 1

            confidence = round(top_score / total_score, 2)
            self._update_mood_state(top_emotion)

            return {
                'top_emotion': top_emotion,
                'scores': filtered,
                'confidence': confidence,
                'mood_state': self.get_mood_state()
            }

        except Exception as e:
            self.logger.error(f"Emotion analysis failed: {str(e)}")
            return {'top_emotion': 'neutral', 'scores': {}, 'confidence': 0.0}

    def _update_mood_state(self, emotion: str):
        """Track mood over time"""
        if emotion != "neutral":
            self.mood_history.append(emotion)

    def get_mood_state(self) -> Dict:
        """Summarize emotional trend over time"""
        if not self.mood_history:
            return {'current_mood': 'neutral', 'history': []}

        trend = {}
        for e in self.mood_history:
            trend[e] = trend.get(e, 0) + 1

        dominant = max(trend, key=trend.get)
        return {'current_mood': dominant, 'history': list(self.mood_history)}


    def get_emotional_response(self, emotion: str) -> str:
        """Convert emotion to empathic response"""
        responses: Dict[str, str] = {
            'happy': "That's wonderful to hear! 😊",
            'sad': "I'm here to help you through this. 💙",
            'angry': "Let's work through this together. 🔥",
            'stressed': "Take a deep breath. We'll handle this step by step. 🌬️",
            'love': "That's beautiful to share. 💖",
            'curious': "Let's explore that together! 🔍",
            'neutral': "Thank you for sharing that. 🌟"
        }
        return responses.get(emotion, "Let's keep talking. 💬")
