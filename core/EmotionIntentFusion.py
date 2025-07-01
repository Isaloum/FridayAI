# =============================================
# File: EmotionIntentFusion.py
# Purpose: Fuse emotion classification + self-intent to assign memory priority
# =============================================

from transformers import pipeline

class EmotionIntentFusion:
    """
    Combines emotion detection and self-intent classification to assess
    emotional salience and guide memory weighting.
    """

    def __init__(self, intent_model, emotion_model_name="j-hartmann/emotion-english-distilroberta-base"):
        # Load HuggingFace emotion classifier (top-1 label)
        self.emotion_classifier = pipeline("text-classification", model=emotion_model_name, top_k=1)
        self.intent_model = intent_model

        self.emotion_weight_map = {
            "joy": 0.3,
            "neutral": 0.1,
            "surprise": 0.5,
            "sadness": 0.85,
            "anger": 0.9,
            "fear": 0.88,
            "disgust": 0.75,
            "love": 0.4
        }

    def analyze(self, text: str):
        """
        Run emotion + intent fusion pipeline.
        Returns:
        {
            emotion: str,
            emotion_weight: float,
            intent_label: str,
            intent_confidence: float,
            memory_priority: float
        }
        """
        emotion_result = self.emotion_classifier(text)[0]
        emotion = emotion_result["label"].lower()
        weight = self.emotion_weight_map.get(emotion, 0.2)

        intent_result = self.intent_model.predict_intent(text)
        intent = intent_result["label"]
        confidence = intent_result["confidence"]

        # Self-directed content increases salience
        priority = round(weight * (1.2 if intent.startswith("self_") else 1.0), 3)

        return {
            "emotion": emotion,
            "emotion_weight": weight,
            "intent_label": intent,
            "intent_confidence": confidence,
            "memory_priority": priority
        }
