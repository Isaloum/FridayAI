# ==============================================================
# File: core/pregnancy/PregnancyWeeklyUpdateAgent.py
# Purpose: Provides weekly pregnancy updates tailored to the user's profile
# ==============================================================

from datetime import datetime

class PregnancyWeeklyUpdateAgent:
    def __init__(self, profile, memory):
        self.profile = profile
        self.memory = memory

    def get_weekly_update(self):
        ctx = self.profile.get_context()
        week = ctx.get("week", 0)
        trimester = ctx.get("trimester", "unknown")

        updates = {
            "first": [
                "Your baby's heart is forming.",
                "Morning sickness is common — rest and hydrate.",
                "Time to schedule your first prenatal check-up."
            ],
            "second": [
                "You might start to feel your baby move soon!",
                "Your energy levels may increase this trimester.",
                "Consider signing up for prenatal yoga."
            ],
            "third": [
                "You’re in the home stretch — rest and prep your space.",
                "Time to pack your hospital bag.",
                "Discuss birth plans and preferences with your provider."
            ]
        }

        summary = updates.get(trimester, ["No specific updates for this stage."])

        return f"🤰 Week {week} — {trimester.title()} Trimester Update:\n- " + "\n- ".join(summary)
