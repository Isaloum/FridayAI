# ================================================
# File: pregnancy_support/core/PregnancySupportCore.py
# Purpose: Emotionally sensitive support for pregnant users
# Includes: Empathy replies + daily emotional check-ins
# ================================================

import os
import json
import random

class PregnancySupportCore:
    def __init__(self):
        self.replies = self._load_empathy_replies()
        self.checkin_prompts = self._load_checkin_prompts()

    def _load_empathy_replies(self):
        try:
            base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "empathy"))
            path = os.path.join(base, "soft_replies.json")
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load empathy replies: {e}")
            return ["I'm here with you, no matter how you feel."]

    def _load_checkin_prompts(self):
        try:
            base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reflection_prompts"))
            path = os.path.join(base, "daily_checkin.txt")
            with open(path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"[ERROR] Failed to load check-in prompts: {e}")
            return ["How are you feeling today?"]

    def respond_to_feeling(self, emotion="neutral"):
        """
        Return a soft empathy reply based on user mood.
        """
        if not self.replies:
            return "You're not alone, and it's okay to feel what you're feeling."
        return random.choice(self.replies)

    def daily_checkin(self):
        """
        Return a daily check-in question for emotional awareness.
        """
        if not self.checkin_prompts:
            return "How are you feeling today?"
        return random.choice(self.checkin_prompts)
