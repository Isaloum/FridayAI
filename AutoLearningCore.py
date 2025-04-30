# AutoLearningCore.py
import re

class AutoLearningCore:
    """Detects facts in user input and saves them automatically."""

    def __init__(self, memory_core):
        self.memory = memory_core

    def process_input(self, user_input):
        """Analyze input and learn new facts if possible."""
        learned = False

        # Very basic fact patterns
        patterns = [
            (r'my name is (.+)', 'your name'),
            (r'i live in (.+)', 'your location'),
            (r'my dog\'s name is (.+)', "your dog's name"),
            (r'my favorite color is (.+)', 'your favorite color'),
            (r'i work at (.+)', 'your workplace')
        ]

        for pattern, key in patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                self.memory.save_fact(key, value)
                learned = True

        return learned
