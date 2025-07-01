# ContextMemory.py
# ----------------
# Tracks recent user/AI exchanges to provide context

class ContextMemory:
    def __init__(self, max_turns=5):
        self.max_turns = max_turns
        self.history = []

    def add_turn(self, user_input, ai_response):
        """
        Store a pair of (user_input, ai_response).
        """
        self.history.append((user_input.strip(), ai_response.strip()))
        if len(self.history) > self.max_turns:
            self.history.pop(0)  # Keep only latest interactions

    def get_context(self):
        """
        Returns a list of recent conversation turns.
        Each turn is a tuple: (user_input, ai_response)
        """
        return self.history

    def summarize_context(self):
        """
        Builds a short summary of past conversation.
        Could be replaced with an LLM later.
        """
        return " | ".join([f"You: {u} / Me: {a}" for u, a in self.history])
