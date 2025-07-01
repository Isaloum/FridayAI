# =====================================
# MemoryReflectionEngine.py - Core Memory-Aware Responder
# =====================================

from typing import Optional, List, Tuple
from difflib import SequenceMatcher

class MemoryReflectionEngine:
    """
    This engine reflects on user input by scanning long-term memory.
    It uses semantic similarity and emotional salience (emotional_weight)
    to return meaningful matches to the current conversation.
    """

    def __init__(self, memory_core, top_k: int = 3):
        """
        :param memory_core: the core memory engine (must expose .memory dict)
        :param top_k: how many top matches to return
        """
        self.memory = memory_core
        self.top_k = top_k

    def reflect(self, user_input: str) -> Optional[List[Tuple[str, float]]]:
        """
        Search memory for values semantically similar to the user input.
        Combine similarity score with emotional_weight to prioritize impactful entries.
        
        :param user_input: current user message
        :return: list of (memory_text, combined_score) tuples
        """
        input_text = user_input.lower()
        matches = []

        for key, versions in self.memory.memory.items():
            for version in versions:
                value = version['value'].lower()
                score = self._similarity(input_text, value)

                if score > 0.45:  # similarity threshold with emotional salience boost
                    weight = version.get("emotional_weight", 0.0)
                    # Combine: favor memories that are both similar and emotionally strong
                    combined_score = round(score * (1 + weight), 4)
                    matches.append((version['value'], combined_score))

        # Sort by combined score (semantic × emotional)
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:self.top_k] if matches else None

    def _similarity(self, a: str, b: str) -> float:
        """
        Use fuzzy string similarity (placeholder for embedding similarity)
        """
        return SequenceMatcher(None, a, b).ratio()

    def reflect_on(self, user_input, reply):
        """
        Combine recent user input and Friday's reply into a memory entry,
        and relate it to emotionally resonant past content if any exists.

        :param user_input: latest user message
        :param reply: Friday's current response
        :return: Reflection string (meta-commentary)
        """
        data = f"{user_input.strip()} → {reply.strip()}"
        similar = self.reflect(user_input)

        if similar:
            best = similar[0]
            return f"You’ve shared similar thoughts before: \"{best[0]}\" (similarity score: {round(best[1]*100)}%)"
        
        # Fallback reflection for emotionally framed sentences
        if "feel" in user_input.lower() or "i am" in user_input.lower():
            return f"It seems like this moment meant something: \"{data}\""
        
        return f"Just noting this for later insight: \"{data}\""
