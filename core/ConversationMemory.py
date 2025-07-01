
# =====================================
# ConversationMemory.py
# =====================================

from collections import deque
from typing import List, Tuple, Optional

class ConversationMemory:
    """
    Manages short-term memory of recent user and AI interactions.
    Enables context awareness and follow-up behavior.
    """

    def __init__(self, max_turns: int = 5):
        self.recent_turns: deque[Tuple[str, str]] = deque(maxlen=max_turns)

    def record(self, user_input: str, ai_response: str):
        """Stores a turn of conversation (user input and AI response)"""
        self.recent_turns.append((user_input, ai_response))

    def last_user_input(self) -> Optional[str]:
        """Returns the most recent user input"""
        if self.recent_turns:
            return self.recent_turns[-1][0]
        return None

    def last_ai_response(self) -> Optional[str]:
        """Returns the most recent AI reply"""
        if self.recent_turns:
            return self.recent_turns[-1][1]
        return None

    def get_context(self) -> List[Tuple[str, str]]:
        """Returns all recent turns of the conversation"""
        return list(self.recent_turns)

    def reset(self):
        """Clears the entire conversation buffer"""
        self.recent_turns.clear()
