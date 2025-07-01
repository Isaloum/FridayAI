# ================================================
# File: EmotionalMemoryFusion.py
# Purpose: Emotionally-tagged memory logging + vector storage
# ================================================

from datetime import datetime

class EmotionalMemoryFusion:
    """
    Logs and fuses user input and AI responses with emotional tags into memory for contextual recall.
    Enhances vector memory with emotional context.
    """

    def __init__(self, vector_memory_core, memory_core, emotion_core):
        self.vector_memory = vector_memory_core
        self.memory = memory_core
        self.emotion_core = emotion_core

    def log_event(self, text: str, source: str = "user"):
        """
        Analyze and store an emotional memory vectorized and tagged.
        """
        emotions = self.emotion_core.analyze(text)
        dominant_emotion = max(emotions, key=emotions.get, default="neutral")

        # Store emotion-tagged memory
        metadata = {
            "source": source,
            "emotion": dominant_emotion,
            "timestamp": datetime.now().isoformat()
        }

        self.vector_memory.ingest(text, metadata=metadata)

        return {
            "text": text,
            "dominant_emotion": dominant_emotion,
            "stored": True
        }
