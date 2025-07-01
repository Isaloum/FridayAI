# ShortTermMemory.py
# -------------------
# Holds the last N user and AI messages

class ShortTermMemory:
    def __init__(self, max_turns=5):
        self.max_turns = max_turns
        self.history = []

    def add_exchange(self, user_input: str, ai_response: str):
        self.history.append({"user": user_input, "friday": ai_response})
        if len(self.history) > self.max_turns:
            self.history.pop(0)

    def get_history(self):
        return self.history

    def clear(self):
        self.history = []
