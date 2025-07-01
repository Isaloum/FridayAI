# ==============================================
# File: core/pregnancy/PregnancySupportCore.py
# Purpose: Pregnancy-specific support module with emotion + memory hooks
# ==============================================

from datetime import datetime
from core.EmotionClassifier import EmotionClassifier
from core.ToneVectorizer import ToneVectorizer
from core.pregnancy.TrimesterLogicUnit import TrimesterLogicUnit
from core.pregnancy.PregnancyMemoryHelper import PregnancyMemoryHelper
from core.pregnancy.PregnancyEmotionPlanner import PregnancyEmotionPlanner
from core.pregnancy.PregnancyJournalInterface import PregnancyJournalInterface
from core.pregnancy.PregnancyNutritionAgent import PregnancyNutritionAgent
from core.pregnancy.PregnancyMentalHealthAgent import PregnancyMentalHealthAgent
from core.pregnancy.PregnancyWeeklyUpdateAgent import PregnancyWeeklyUpdateAgent
import re


def is_meaningful(text):
    """Detects gibberish or meaningless input (e.g. random letters)."""
    text = text.strip().lower()
    if not text or len(text) < 3:
        return False
    if re.fullmatch(r'[a-z]{1,3}', text):
        return False  # e.g. "aaa", "wer", "hj"
    if len(text.split()) == 1 and not re.search(r'[aeiou]', text):
        return False  # vowel-less gibberish
    if re.search(r'[0-9]{3,}', text):
        return False  # contains large number strings
    if re.search(r'[a-z]{5,}', text) and not re.search(r'\s', text):
        return False  # long alphabetic junk without space
    return True


class PregnancySupportCore:
    def __init__(self, memory, emotion_core, identity, profile):
        # Core dependencies
        self.memory = memory
        self.emotion_core = emotion_core
        self.emotion_classifier = EmotionClassifier()
        self.identity = identity
        self.profile = profile

        # Domain-specific modules
        self.memory_logger = PregnancyMemoryHelper(memory, identity)
        self.journal = PregnancyJournalInterface(memory, emotion_core, identity)
        self.tone_vectorizer = ToneVectorizer()
        self.nutrition_agent = PregnancyNutritionAgent(profile)
        self.mental_health_agent = PregnancyMentalHealthAgent(memory, profile)
        self.weekly_agent = PregnancyWeeklyUpdateAgent(profile, memory)
        from core.ToneRewriterCore import ToneRewriterCore
        self.tone = ToneRewriterCore()
        from core.HumanConversationCore import HumanConversationCore
        self.convo = HumanConversationCore()

    def _log_goals_if_needed(self):
        # Check if trimester goals are already logged for today
        trimester = self.profile.get_context().get("trimester")
        if not trimester:
            return
        existing = self.memory.get_recent_entries(entry_type="pregnancy_goals", days=1)
        if any(e.get("trimester") == trimester for e in existing):
            return
        # Map trimester to suggested goals
        goal_map = {
            "first": [
                "Schedule your first prenatal checkup.",
                "Start a gentle walking routine.",
                "Begin a food/mood diary."
            ],
            "second": [
                "Start gentle stretching or yoga.",
                "Document positive body changes.",
                "Discuss support systems with partner."
            ],
            "third": [
                "Pack your hospital bag.",
                "Practice breathing techniques.",
                "Log things that bring you comfort."
            ]
        }
        # Save new goals to memory
        self.memory.save_memory({
            "type": "pregnancy_goals",
            "trimester": trimester,
            "goals": goal_map.get(trimester, []),
            "timestamp": datetime.utcnow().isoformat()
        })

    def update_trimester(self, weeks: int):
        # Determine and store the user's current trimester based on weeks
        trimester = TrimesterLogicUnit.get_trimester(weeks)
        self.profile.update_profile(weeks=weeks, trimester=trimester)
        return trimester

    def respond_to_feeling(self, feeling: str):
        # Validate the feeling text for meaningful input
        if not is_meaningful(feeling):
            return "[🛑 Input rejected as gibberish by is_meaningful()]\nI'm not sure how to interpret that. Want to try describing how you feel another way?"

        # Analyze emotion from input
        profile = self.emotion_classifier.analyze(feeling)
        emotion = profile.get('top_emotion') or 'neutral'
        certainty = profile.get('certainty', 0.0)

        # Fallback if emotion detection failed or low confidence
        if emotion == "unknown" or certainty < 0.3:
            emotion = "neutral"
            certainty = 0.0

        # Log emotion and feeling
        self.identity.log_event(f"Pregnancy feeling: {feeling}", mood=emotion, source="pregnancy")
        self.memory_logger.log_event(feeling, emotion, self.profile.get_context().get("trimester", None))

        # Anchor negative emotions in memory
        if emotion in ["sadness", "anxious", "resentful"]:
            self.memory.save_memory({
                "type": "anchor_event",
                "emotion": emotion,
                "text": feeling,
                "trimester": self.profile.get_context().get("trimester", None),
                "timestamp": datetime.utcnow().isoformat()
            })

        # Log trimester-specific goals if needed
        self._log_goals_if_needed()

        # Recall recent emotion memory
        recent = self.memory_logger.recall_recent_emotions(limit=3)
        if recent:
            history = "\n".join([f"- {e.capitalize()} from: '{t}'" for e, t in recent])
            memory_blurb = f"\n\n📓 You've recently felt:\n{history}"
        else:
            memory_blurb = ""

        # Analyze emotional drift from recent pregnancy logs
        recent = self.memory.get_recent_entries(entry_type="pregnancy_log", days=7)[-3:]
        recent_emotions = [e.get("emotion") for e in recent if e.get("emotion")]
        drift_prompt = None
        if recent_emotions and emotion != recent_emotions[0]:
            dominant = max(set(recent_emotions), key=recent_emotions.count)
            if dominant != emotion:
                drift_prompt = f"🧭 You've felt more *{dominant}* recently. Is today feeling different?"

        # Generate and rephrase final response
        raw_reply = self.convo.reply(feeling)
        soft_reply = self.tone.rewrite(raw_reply, tone="reassure")
        if drift_prompt:
            soft_reply = drift_prompt + "\n\n" + soft_reply
        soft_reply += memory_blurb
        soft_reply += f"\n\n[🧠 Raw: {raw_reply}]"
        soft_reply += "\n💬 Want this in a calmer tone? Type: tone:calm"

        return soft_reply

    def compassionate_reply(self, emotion: str, feeling: str) -> str:
        # Tailored compassion replies based on context and emotion
        ctx = self.profile.get_context()
        flags = ctx.get("flags", {})

        if flags.get("IVF") and emotion == "anxious":
            return "That anxious feeling is real — especially on an IVF path. I'm walking this with you."

        if flags.get("high_risk") and emotion in ["sad", "anxious"]:
            return "With a high-risk label, every emotion carries more weight. Let’s unpack it, gently."

        replies = {
            "sad": f"That sadness you're feeling — do you want to sit with it together for a moment?",
            "happy": f"That joy? It matters. Want to share what sparked it?",
            "neutral": f"I’m here. Even quiet feelings deserve attention. Want to talk?",
            "angry": f"It’s okay to feel that fire. What’s under it?",
            "anxious": f"Your mind is racing. Want to breathe and sort it out together?"
        }
        return replies.get(emotion, "I'm here with you. No pressure. Just presence.")

    def trimester_insight(self):
        # Return current trimester awareness
        trimester = self.profile.get_context().get("trimester", "unknown")
        return f"You are currently in your {trimester} trimester. Every stage is unique — your needs evolve, and I’m adapting with you."

    def _soothing_reply(self, emotion: str):
        # Map emotion to calming response
        responses = {
            "anxious": "It’s okay to feel anxious. Let’s take a slow breath together.",
            "tired": "Fatigue is valid. Rest matters. I’ll go easy with you.",
            "overwhelmed": "One step at a time — I’m right here with you.",
            "happy": "Joy during pregnancy is sacred. Let's capture this feeling.",
            "neutral": "I’m by your side, no matter the feeling."
        }
        return responses.get(emotion, "That’s valid. Would you like to talk more about it?")

    def suggest_emotion_plan(self, feeling: str):
        # Propose action plan based on emotion
        profile = self.emotion_classifier.analyze(feeling)
        emotion = profile["top_emotion"]
        return PregnancyEmotionPlanner.generate_plan(emotion)

    def suggest_nutrition(self):
        # Recommend dietary advice
        return self.nutrition_agent.get_nutrition_tips()

    def suggest_mental_health(self):
        # Offer wellness practices
        return self.mental_health_agent.get_support_plan()

    def generate_weekly_update(self):
        # Return a personalized weekly summary
        return self.weekly_agent.build_update()
