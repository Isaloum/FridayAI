# SelfQueryingCore.py

class SelfQueryingCore:
    """When Friday doesn't know something, he asks back to learn."""

    def __init__(self, memory_core):
        self.memory = memory_core

    def ask_for_missing_info(self, user_input):
        """If key info is missing from memory, ask the user."""
        questions = {
            "your name": "What should I call you?",
            "your location": "Where are you located?",
            "your dog's name": "What is your dog's name?",
            "your favorite color": "What's your favorite color?",
            "your workplace": "Where do you work?"
        }

        location_phrases = ["where am i", "my location", "where am i located", "current location"]
        if "your location" not in self.memory.memory:
            if any(phrase in user_input for phrase in location_phrases):
                return "Where are you located?"

        for key in questions:
            if key not in self.memory.memory:
                if key in user_input or any(word in user_input for word in key.split()):
                    return questions[key]

        return None
 