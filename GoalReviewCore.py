# GoalReviewCore.py
# ------------------------------
# Provides introspective analysis and updates for FridayAI's long-term goals.
# Links with LongTermIntentCore to offer reviews, prompt for updates, and flag drift.

from datetime import datetime, timedelta

class GoalReviewCore:
    def __init__(self, intent_core):
        self.intent_core = intent_core

    def summarize_goals(self):
        """Returns a high-level summary of all active long-term goals."""
        all_goals = self.intent_core.get_all_intents()
        if not all_goals:
            return "You have no active long-term goals."

        summary = ["Here’s what we’re working toward:"]
        for goal_id, data in all_goals.items():
            summary.append(f"- {data['description']} [{data['priority']}] :: {data['category']}")
        return "\n".join(summary)

    def find_stale_goals(self, days=14):
        """Detects goals that haven’t been updated in the last X days."""
        stale = []
        threshold = datetime.now() - timedelta(days=days)
        for goal_id, data in self.intent_core.get_all_intents().items():
            last_updated = datetime.fromisoformat(data["updated_at"])
            if last_updated < threshold:
                stale.append(f"⚠️ {data['description']} — last touched {last_updated.date()}")
        return stale

    def recommend_updates(self):
        """Suggests which goals should be reviewed or revised."""
        stale_goals = self.find_stale_goals()
        if not stale_goals:
            return "All goals are relatively up-to-date."
        return "Some goals may need attention:\n" + "\n".join(stale_goals)

    def review_emotion_alignment(self, emotion_profile):
        """Checks if current emotions mismatch previously tagged goals."""
        misaligned = []
        for goal_id, data in self.intent_core.get_all_intents().items():
            tags = data.get("emotion_tags", [])
            if tags and emotion_profile not in tags:
                misaligned.append(f"🔄 {data['description']} — tagged with {tags}, but you're feeling {emotion_profile}.")
        if not misaligned:
            return "All goals match your current emotional tone."
        return "Emotionally misaligned goals:\n" + "\n".join(misaligned)

# Example usage (test mode)
if __name__ == "__main__":
    from LongTermIntentCore import LongTermIntentCore

    intent_core = LongTermIntentCore()
    reviewer = GoalReviewCore(intent_core)

    print(reviewer.summarize_goals())
    print("\n" + reviewer.recommend_updates())
    print("\n" + reviewer.review_emotion_alignment("frustrated"))
