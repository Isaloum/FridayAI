# EmotionCore.py
import logging
from typing import Dict

class EmotionCore:
    """Advanced emotional state detection system with fallback mechanisms"""
    
    def __init__(self):
        self.logger = logging.getLogger("FridayAI.EmotionCore")
        self.emotion_matrix = {
            "happy": ["happy", "joy", "excited", "great", "awesome", "fantastic"],
            "sad": ["sad", "depressed", "unhappy", "crying", "miserable"],
            "angry": ["angry", "mad", "furious", "annoyed", "pissed"],
            "stressed": ["stressed", "overwhelmed", "anxious", "pressure"],
            "love": ["love", "adore", "care", "affection", "cherish"],
            "curious": ["curious", "wonder", "why", "how", "explain"],
            "neutral": []
        }

    def analyze(self, text: str) -> str:
        """
        Analyze text for emotional content with confidence scoring
        Returns: Detected emotion or 'neutral' if uncertain
        """
        try:
            if not isinstance(text, str) or not text.strip():
                return "neutral"
            
            clean_text = text.lower().strip()
            emotion_scores = {emotion: 0 for emotion in self.emotion_matrix}

            # Score each emotion based on keyword matches
            for emotion, keywords in self.emotion_matrix.items():
                emotion_scores[emotion] = sum(
                    1 for keyword in keywords 
                    if f" {keyword} " in f" {clean_text} "  # Whole word matching with padding
                )

            # Get emotion with highest score
            max_score = max(emotion_scores.values())
            return (
                max((emotion for emotion, score in emotion_scores.items() 
                    if score == max_score), key=lambda x: len(self.emotion_matrix[x]))
                if max_score > 0
                else "neutral"
            )

        except Exception as e:
            self.logger.error(f"Emotion analysis failed: {str(e)}")
            return "neutral"

    def get_emotional_response(self, emotion: str) -> str:
        """Convert detected emotion to appropriate response"""
        responses: Dict[str, str] = {
            'happy': "That's wonderful to hear! 😊",
            'sad': "I'm here to help you through this. 💙",
            'angry': "Let's work through this together. 🔥",
            'stressed': "Take a deep breath. We'll handle this step by step. 🌬️",
            'love': "That's beautiful to share. 💖",
            'curious': "Let's explore that together! 🔍",
            'neutral': "Thank you for sharing that. 🌟"
        }
        return responses.get(emotion, "Let's continue our conversation. 💬")