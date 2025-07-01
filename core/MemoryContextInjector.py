# =============================================
# File: core/MemoryContextInjector.py
# Purpose: Inject relevant memories into conversation context
# Location: core/ folder
# =============================================

from datetime import datetime, timedelta

class MemoryContextInjector:
    """
    Finds relevant memories and adds them to conversation context.
    Makes Friday remember past conversations.
    """
    
    def __init__(self, memory_core):
        """
        Start the memory injector.
        Needs access to memory to search old conversations.
        """
        self.memory = memory_core
       #print("[DEBUG] MemoryContextInjector initialized")
    
    def inject_context(self, user_input: str, max_memories: int = 3) -> dict:
        """
        Find memories related to user input.
        Return context to help Friday remember.
        """
        # Get recent memories from the last week
        recent_memories = self.memory.get_recent_entries(days=7)
        
        # Find memories that might be related to current input
        relevant = []
        input_words = user_input.lower().split()
        
        for memory in recent_memories:
            memory_text = memory.get("text", "").lower()
            
            # Check if any words from input appear in memory
            for word in input_words:
                if len(word) > 3 and word in memory_text:  # Only check meaningful words
                    relevant.append(memory)
                    break
        
        # Limit to most recent relevant memories
        relevant = relevant[-max_memories:]
        
        return {
            "relevant_memories": relevant,
            "memory_count": len(relevant),
            "reflection": f"Found {len(relevant)} related memories"
        }


def inject(user_input: str) -> dict:
    """
    Simple function that other parts of Friday can use.
    For now, returns basic context without real memory search.
    """
    return {
        "reflection": f"Processing: {user_input[:50]}...",
        "relevant_memories": [],
        "memory_count": 0
    }