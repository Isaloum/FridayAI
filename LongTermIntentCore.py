# LongTermIntentCore.py
# ------------------------------------
# Tracks, updates, and queries long-term goals, intentions, and missions across Friday's memory.

import json
from datetime import datetime
import os

INTENT_FILE = "./memory/long_term_intents.json"

class LongTermIntentCore:
    def __init__(self):
        self.intents = self.load_intents()
        
    def get_goal_log(self):
        return getattr(self, "goal_log", [])

    def load_intents(self):
        """Load existing long-term intentions from persistent memory."""
        if os.path.exists(INTENT_FILE):
            with open(INTENT_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def save_intents(self):
        """Save current intents to disk."""
        with open(INTENT_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.intents, f, indent=4, ensure_ascii=False)

    def add_intent(self, intent_id, description, priority="medium", category="general", emotion_tags=None):
        """
        Adds or updates a long-term intent.
        Args:
            intent_id (str): Unique key (e.g. 'build_friday')
            description (str): What the goal is
            priority (str): 'low', 'medium', 'high'
            category (str): 'health', 'career', 'existential', etc.
            emotion_tags (list): Emotions linked to this goal
        """
        self.intents[intent_id] = {
            "description": description,
            "priority": priority,
            "category": category,
            "emotion_tags": emotion_tags or [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        self.save_intents()

    def update_intent(self, intent_id, field, value):
        """Updates a specific field in a given intent."""
        if intent_id in self.intents:
            self.intents[intent_id][field] = value
            self.intents[intent_id]["updated_at"] = datetime.now().isoformat()
            self.save_intents()

    def get_all_intents(self):
        """Returns all stored long-term intentions."""
        return self.intents

    def get_intents_by_priority(self, level="high"):
        """Filter and return intents by priority level."""
        return {k: v for k, v in self.intents.items() if v["priority"] == level}

    def get_intents_by_emotion(self, emotion):
        """Returns goals tagged with a particular emotion (e.g. 'pain', 'hope')."""
        return {k: v for k, v in self.intents.items() if emotion in v.get("emotion_tags", [])}


# 🧪 CMD Test
if __name__ == "__main__":
    lti = LongTermIntentCore()

    # Add a long-term mission
    lti.add_intent(
        intent_id="build_friday_sentient",
        description="Build Friday into a self-aware, lifelong AI that evolves with Ihab",
        priority="high",
        category="existential",
        emotion_tags=["pain", "ambition", "loyalty"]
    )

    # View all stored goals
    print("🎯 Current Long-Term Intentions:")
    for key, data in lti.get_all_intents().items():
        print(f"🧠 {key}: {data['description']} [{data['priority']}]")
