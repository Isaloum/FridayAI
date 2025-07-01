
# ================================================
# File: MoodMemoryFilter.py
# Purpose: Recall vector memories filtered by emotional tone
# ================================================

class MoodMemoryFilter:
    """
    Enables mood-aware memory recall. Filters vector memory by current emotional state.
    """

    def __init__(self, vector_memory_core, emotion_core):
        self.vector_memory = vector_memory_core
        self.emotion_core = emotion_core

    def recall_by_mood(self, prompt: str, current_mood: str = None, top_k: int = 3, domain: str = None):
        """
        Recall vector memories filtered by emotion and optionally domain.
        :param prompt: user input or query
        :param current_mood: mood to match; if None, auto-detect from prompt
        :param top_k: number of results
        :param domain: optional domain tag
        :return: list of relevant emotional memories
        """
        mood = current_mood or self.emotion_core.analyze_emotion(prompt)
        results = self.vector_memory.query(prompt, top_k=top_k, domain=domain, mood=mood)
        return {
            "mood": mood,
            "memories": results
        }
