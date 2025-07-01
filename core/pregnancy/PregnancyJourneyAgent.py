# ==============================================
# File: core/pregnancy/PregnancyJourneyAgent.py
# Purpose: Send weekly updates and repurpose content for blog/email/family
# ==============================================

from datetime import datetime

class PregnancyJourneyAgent:
    def __init__(self, profile, memory):
        self.profile = profile
        self.memory = memory

    def journey_update(self):
        if not hasattr(self.profile, "get_context") or not self.profile.get_context().get("week"):
            return "⚠️ Week info not set. Journey update skipped."

        ctx = self.profile.get_context()
        week = ctx.get("week", 0)
        trimester = ctx.get("trimester", "unknown")

        summary = f"🤰 Week {week} ({trimester} trimester):\n"
        summary += "- Update: Baby is growing fast.\n"
        summary += "- Tip: Stay hydrated.\n"
        summary += "- Reminder: Log how you feel today.\n"

        self.memory.save_memory({
            "type": "journey_update",
            "week": week,
            "trimester": trimester,
            "text": summary,
            "timestamp": datetime.utcnow().isoformat()
        })

        return summary

    def for_blog(self):
        return f"Week {self.profile.get_context().get('week')}: A new chapter unfolds..."

    def for_email(self):
        return f"Hey there — here’s your weekly check-in for week {self.profile.get_context().get('week')}!"

    def for_family(self):
        return f"Family Update — Week {self.profile.get_context().get('week')} is here!"
