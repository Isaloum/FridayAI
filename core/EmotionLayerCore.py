# EmotionLayerCore.py – Multidimensional Emotion Detector for FridayAI

import re
from collections import defaultdict

class EmotionLayerCore:
    def __init__(self):
        self.emotion_map = {
            "happy":     ["joy", "grateful", "thankful", "excited", "relieved", "blessed"],
            "sad":       ["down", "alone", "heartbroken", "depressed", "tired", "empty"],
            "angry":     ["pissed", "furious", "mad", "rage", "hate", "annoyed"],
            "anxious":   ["worried", "nervous", "scared", "panic", "overthinking", "uncertain"],
            "hopeful":   ["hope", "wish", "believe", "faith", "waiting", "dream"],
            "guilty":    ["sorry", "regret", "ashamed", "my fault", "blame"],
            "lonely":    ["alone", "ignored", "unseen", "forgotten", "isolated"],
            "resentful": ["used", "taken for granted", "again", "never noticed"],
            "conflicted": ["torn", "mixed", "don’t know", "should I", "but"]
        }

    def detect_emotion_layers(self, text: str) -> dict:
        text = text.lower()
        emotion_scores = defaultdict(float)

        for emotion, keywords in self.emotion_map.items():
            for word in keywords:
                if re.search(rf"\\b{re.escape(word)}\\b", text):
                    emotion_scores[emotion] += 1

        # Normalize scores (scale to max 1)
        max_score = max(emotion_scores.values(), default=1)
        for k in emotion_scores:
            emotion_scores[k] = round(emotion_scores[k] / max_score, 2)

        # Select primary + supporting
        sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)

        # 🚫 Filter out low-confidence emotion triggers
        if not sorted_emotions or sorted_emotions[0][1] < 0.6:
            return {
                "primary": "neutral",
                "supporting": [],
                "raw_scores": dict(emotion_scores)
            }

        primary = sorted_emotions[0][0]
        supporting = [e[0] for e in sorted_emotions[1:3] if e[1] > 0]

        return {
            "primary": primary,
            "supporting": supporting,
            "raw_scores": dict(emotion_scores)
        }
