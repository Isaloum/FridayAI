# ==============================================
# File: core/pregnancy/PregnancyProfile.py
# Purpose: Holds user-specific pregnancy context for personalization
# ==============================================

from datetime import datetime, timedelta

class PregnancyProfile:
    def __init__(self, due_date: str = None, custom_notes: str = ""):
        self.due_date = due_date  # Expected in 'YYYY-MM-DD' format
        self.created_at = datetime.now().isoformat()
        self.custom_notes = custom_notes
        self.flags = {
            "IVF": False,
            "high_risk": False,
            "first_pregnancy": True
        }
        self.preferences = {
            "diet": "general",
            "birth_plan": "hospital",
            "language_tone": "gentle"
        }

    def current_week(self):
        if not self.due_date:
            return None
        due = datetime.strptime(self.due_date, "%Y-%m-%d")
        today = datetime.today()
        conception = due - timedelta(weeks=40)
        return max(1, (today - conception).days // 7)

    def summary(self):
        return {
            "weeks": self.current_week(),
            "flags": self.flags,
            "preferences": self.preferences,
            "notes": self.custom_notes
        }
