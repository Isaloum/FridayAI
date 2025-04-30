# EmotionCore.py
# Handles emotional detection for True Friday

class EmotionCore:
    """Detects emotional tone from user input."""

    def __init__(self):
        # Simple emotion keyword maps
        self.emotion_map = {
            "sad": ["sad", "depressed", "unhappy", "crying", "lost", "lonely"],
            "happy": ["happy", "excited", "joyful", "laughing", "glad"],
            "angry": ["angry", "mad", "furious", "annoyed", "frustrated"],
            "stressed": ["stressed", "overwhelmed", "anxious", "nervous", "worried"],
            "love": ["love", "like", "adore", "care about"],
            "sick": ["sick", "ill", "pain", "hurt", "injured"]
        }

    def detect_emotion(self, text):
        """Detects user's emotion based on keywords."""
        text = text.lower()
        detected_emotions = []

        for emotion, keywords in self.emotion_map.items():
            if any(word in text for word in keywords):
                detected_emotions.append(emotion)

        if not detected_emotions:
            return "neutral"
        return detected_emotions[0]  # Return first detected emotion for now
