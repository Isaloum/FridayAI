# ======================================
# File: ConversationFlowController.py
# Purpose: Enhances Friday's conversational flow across turns
# Adjusts tone, follow-ups, and emotional carry-over
# ======================================

class ConversationFlowController:
    def __init__(self):
        self.last_emotion = None
        self.last_topic = None
        self.turn_count = 0

    def track_context(self, user_input, emotion_summary):
        self.last_emotion = emotion_summary
        self.last_topic = self._detect_topic(user_input)
        self.turn_count += 1

    def _detect_topic(self, user_input):
        keywords = ["job", "love", "family", "future", "feeling", "goal", "dream"]
        for word in keywords:
            if word in user_input.lower():
                return word
        return "general"

    def suggest_follow_up(self):
        if self.last_topic == "general":
            return "Want to keep going with that thought, or pivot to something lighter?"

        if self.last_emotion in ["sad", "anxious"]:
            return "Want to unpack that more, or just breathe together for a moment?"

        if self.last_emotion == "curious":
            return "Should I explain more, or do you want to explore a different angle?"

        return "I'm still with you — want to keep that thread going?"

    def inject_flow(self, base_reply):
        # Optional: add flow follow-up based on tracked state
        if self.turn_count >= 2:
            follow_up = self.suggest_follow_up()
            return f"{base_reply} {follow_up}"
        return base_reply
