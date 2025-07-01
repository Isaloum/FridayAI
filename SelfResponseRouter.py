# =============================================
# File: SelfResponseRouter.py
# Purpose: Route self-directed user questions to Friday's internal systems
# =============================================

class SelfResponseRouter:
    """
    Routes self-directed questions to Friday's introspective subsystems:
    - self_mood → SelfNarrativeCore
    - self_identity → NarrativeMemoryFusion
    - self_memory → MemoryReflectionEngine
    - self_behavior → SelfAwarenessCore
    """

    def __init__(self, intent_model, narrative, awareness, memory_reflector, identity_fusion):
        self.intent_model = intent_model
        self.narrative = narrative
        self.awareness = awareness
        self.memory_reflector = memory_reflector
        self.identity_fusion = identity_fusion

    def route(self, user_input: str, emotion: str = "neutral") -> str:
        """
        Determine intent label and return a self-reflective response from the correct module.
        """
        result = self.intent_model.predict_intent(user_input)
        label = result["label"]
        confidence = result["confidence"]

        if confidence < 0.6:
            return "I'm not entirely sure what you're asking about me, but I'm here."

        if label == "self_mood":
            mood = self.narrative.get_raw_state().get("mood", "neutral")
            return f"I’d describe my current mood as: {mood}."

        elif label == "self_identity":
            return self.identity_fusion.summarize_identity()

        elif label == "self_memory":
            return self.memory_reflector.reflect_on(user_input, "[REDACTED]")

        elif label == "self_behavior":
            return self.awareness.generate_self_reflection()

        return "That sounds like something I should think more about. Let me reflect on it."
