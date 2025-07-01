"""
NLUProcessor.py
-------------------
This module adds Natural Language Understanding to FridayAI.
It tries to detect what the user is trying to say,
even when the sentence is unclear, short, or coded.

It supports:
- Intent classification (e.g., "calorie_calc", "emotion_check")
- Entity extraction (e.g., grams of food, keywords)
"""

import re

class NLUProcessor:
    def __init__(self):
        # Define known food calories per gram (simple example)
        self.calorie_map = {
            "egg": 1.55,     # per gram
            "pb": 5.9,       # peanut butter
            "banana": 0.89,
            "rice": 1.3,
        }

    def parse(self, text: str) -> dict:
        """
        Main method to detect intent and extract meaning.
        Returns a dictionary with:
        - intent
        - entities (e.g., food items, emotions)
        """
        lowered = text.lower().strip()

        # 1. Emotion check intent
        if lowered in ["r u okay", "are you okay", "you good", "what's up"]:
            return {"intent": "emotion_check", "entities": {}}

        # 2. Calorie calculation intent
        food_matches = re.findall(r'(\d+)g\s*(\w+)', lowered)
        if food_matches:
            total_calories = 0
            foods = {}
            for grams, food in food_matches:
                food_key = food.lower()
                g = int(grams)
                if food_key in self.calorie_map:
                    cal = g * self.calorie_map[food_key]
                    foods[food_key] = {"grams": g, "calories": cal}
                    total_calories += cal

            return {
                "intent": "calorie_calc",
                "entities": {
                    "foods": foods,
                    "total_calories": round(total_calories, 2)
                }
            }

        # 3. Unknown input → fallback
        return {"intent": "unknown", "entities": {}}


# -------------------
# TEST (run this file directly)
# -------------------
if __name__ == "__main__":
    nlu = NLUProcessor()

    examples = [
        "R U okay",
        "4g egg + 5g PB = Cal?",
        "10g rice and 15g banana",
        "who is Elon"
    ]

    for text in examples:
        print(f"\nUser said: {text}")
        result = nlu.parse(text)
        print("Intent:", result["intent"])
        print("Entities:", result["entities"])
