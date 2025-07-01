# ================================================
# File: BeliefDriftSimulator.py
# Purpose: Simulate belief evolution based on mood + memory
# ================================================

from datetime import datetime

class BeliefDriftSimulator:
    """
    Simulates long-term belief drift based on emotional patterns and repeated memory exposures.
    """

    def __init__(self, identity_core, emotion_core, vector_memory_core):
        self.identity = identity_core
        self.emotion_core = emotion_core
        self.vector_memory = vector_memory_core

    def simulate_drift(self, user_input: str):
        """
        Evaluate emotional tone and memory impact of user input to trigger identity updates.
        """
        # Detect emotional tone
        mood = self.emotion_core.analyze_emotion(user_input)

        # Recall matching memories
        memory_hits = self.vector_memory.query(user_input, top_k=3, mood=mood)

        # Build a reflection sentence
        reflection = f"Today, the user said something {mood}. It reminded me of:\n"
        for mem in memory_hits:
            reflection += f"- {mem['text']} (emotion: {mem['metadata'].get('emotion', 'unknown')})\n"

        # Inject into identity
        if hasattr(self.identity, "log_event"):
            self.identity.log_event(reflection.strip(), kind="drift")
        if hasattr(self.identity, "update_mood"):
            self.identity.update_mood(mood)

        return {
            "mood": mood,
            "reflected": reflection.strip(),
            "memory_count": len(memory_hits),
            "timestamp": datetime.now().isoformat()
        }
