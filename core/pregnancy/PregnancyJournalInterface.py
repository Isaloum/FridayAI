# ==================================================
# File: core/pregnancy/PregnancyJournalInterface.py
# Purpose: Guided journal for daily pregnancy reflection
# ==================================================

from core.pregnancy.PregnancyEmotionPlanner import PregnancyEmotionPlanner
from core.pregnancy.PregnancyMemoryHelper import PregnancyMemoryHelper
from core.pregnancy.TrimesterLogicUnit import TrimesterLogicUnit

class PregnancyJournalInterface:
    def __init__(self, memory_core, emotion_core, identity_core):
        self.memory_logger = PregnancyMemoryHelper(memory_core, identity_core)
        self.emotion_core = emotion_core
        self.identity = identity_core
        self.trimester = None

    def set_weeks(self, weeks: int):
        self.trimester = TrimesterLogicUnit.get_trimester(weeks)

    def prompt_and_log(self):
        print("📝 How are you feeling today during pregnancy?\n")
        entry = input("> ").strip()
        emotion = self.emotion_core.analyze_emotion(entry).get("top_emotion", "neutral")

        plan = PregnancyEmotionPlanner.generate_plan(emotion)
        self.memory_logger.log_event(entry, emotion, self.trimester)

        print(f"\n🧠 Detected Emotion: {emotion}")
        print("🩺 Suggested Self-Care Steps:")
        for step in plan:
            print(f"- {step}")


# ==========================
# CLI Test Mode
# ==========================
if __name__ == "__main__":
    from core.MemoryCore import MemoryCore
    from core.EmotionCoreV2 import EmotionCoreV2
    from core.SelfNarrativeCore import SelfNarrativeCore

    memory = MemoryCore()
    emotion = EmotionCoreV2()
    identity = SelfNarrativeCore()

    journal = PregnancyJournalInterface(memory, emotion, identity)
    
    print("📆 Daily Pregnancy Journal")
    weeks = int(input("How many weeks pregnant?\n> "))
    journal.set_weeks(weeks)

    while True:
        journal.prompt_and_log()
        again = input("\nAnother entry? (y/n)\n> ")
        if again.lower() != "y":
            break
