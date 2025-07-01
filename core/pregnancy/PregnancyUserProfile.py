# ==================================================
# File: core/pregnancy/PregnancyUserProfile.py
# Purpose: Stores and manages user's personalized pregnancy context
# ==================================================

class PregnancyUserProfile:
    def __init__(self):
        self.week = None
        self.health_flags = []
        self.preferences = {
            "birth_plan": None,
            "mental_health_focus": False,
            "nutrition_focus": False
        }

    def set_week(self, week: int):
        self.week = week

    def add_health_flag(self, flag: str):
        if flag not in self.health_flags:
            self.health_flags.append(flag)

    def update_preference(self, key: str, value):
        if key in self.preferences:
            self.preferences[key] = value

    def get_context(self):
        return {
            "week": self.week,
            "trimester": self._get_trimester(),
            "health_flags": self.health_flags,
            "preferences": self.preferences
        }

    def _get_trimester(self):
        if self.week is None:
            return None
        if self.week < 13:
            return "first"
        elif self.week < 28:
            return "second"
        return "third"

    def update_profile(self, weeks=None, trimester=None, conditions=None, preferences=None):
        if weeks is not None:
            self.set_week(weeks)
        if conditions:
            for flag in conditions:
                self.add_health_flag(flag)
        if preferences:
            for k, v in preferences.items():
                self.update_preference(k, v)
