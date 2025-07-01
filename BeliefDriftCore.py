# BeliefDriftCore.py
# --------------------------------------------
# Detects contradictions, neglect, and drift in FridayAI's belief system.
# Challenges outdated or forgotten beliefs and suggests reassessment.

from datetime import datetime, timedelta

class BeliefDriftCore:
    def __init__(self, intent_core, emotion_core):
        self.intent_core = intent_core
        self.emotion_core = emotion_core

    def scan_for_drift(self, emotion_profile: str, days_stale: int = 21):
        """
        Identifies beliefs/goals that haven't been emotionally reinforced or updated recently.
        Returns a list of belief drift alerts.
        """
        drift_alerts = []
        threshold = datetime.now() - timedelta(days=days_stale)
        all_intents = self.intent_core.get_all_intents()

        for goal_id, data in all_intents.items():
            updated_at = datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat()))
            emotion_tags = data.get("emotion_tags", [])

            if updated_at < threshold:
                drift_alerts.append(f"⚠️ '{data['description']}' hasn't been revisited since {updated_at.date()}.")

            elif emotion_profile not in emotion_tags:
                drift_alerts.append(f"⚖️ '{data['description']}' is tagged with {emotion_tags}, but your current tone is '{emotion_profile}'.")

        return drift_alerts

    def summarize_drift_status(self, emotion_profile: str) -> str:
        """
        Generates a formatted report on all current drifts detected.
        """
        drift = self.scan_for_drift(emotion_profile)
        if not drift:
            return "✅ All beliefs are consistent with your current mood and recent behavior."

        report = ["🧠 Belief Drift Detected:"] + drift
        return "\n".join(report)

# 🧪 Test mode
if __name__ == "__main__":
    from LongTermIntentCore import LongTermIntentCore
    from EmotionCore import EmotionCore

    icore = LongTermIntentCore()
    ecore = EmotionCore()

    bdc = BeliefDriftCore(icore, ecore)
    print(bdc.summarize_drift_status("reflective"))
