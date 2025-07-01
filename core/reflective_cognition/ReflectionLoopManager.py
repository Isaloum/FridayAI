# =============================================
# File: ReflectionLoopManager.py
# Purpose: Periodic, introspective self-awareness cycle (Option B)
# =============================================

from datetime import datetime, timedelta
from core.reflective_cognition.SelfAwarenessEngine import SelfAwarenessEngine
from core.memory_bank.SelfNarrativeCore import SelfNarrativeCore
from core.MemoryCore import MemoryCore
from core.EmotionCoreV2 import EmotionCoreV2
from core.goal_logic.GoalGenerator import GoalGenerator
from tools.goal_to_task_converter import GoalToTaskConverter

class ReflectionLoopManager:
    """
    This loop orchestrates Friday's deeper self-awareness cycle.
    It runs periodically, analyzing behavior, logging introspection,
    and triggering belief updates + long-term goals if needed.
    """

    def __init__(self, memory, emotion_core, belief_core=None, interval_hours=24):
        """
        :param memory: MemoryCore (used in SelfAwarenessEngine)
        :param emotion_core: EmotionCoreV2 (used in SelfAwarenessEngine)
        :param belief_core: BeliefDriftCore (optional)
        :param interval_hours: frequency of self-reflection
        """
        self.memory = memory
        self.emotion_core = emotion_core
        self.narrative = SelfNarrativeCore()
        self.self_awareness = SelfAwarenessEngine(emotion_core, memory)
        self.goal_generator = GoalGenerator()
        self.goal_to_task = GoalToTaskConverter()
        self.belief_core = belief_core
        self.interval = timedelta(hours=interval_hours)
        self.last_run = None

    def should_run(self) -> bool:
        now = datetime.now()
        return self.last_run is None or (now - self.last_run) >= self.interval

    def run_reflection_cycle(self):
        if not self.should_run():
            return

        print("🧠 [ReflectionLoop] Starting self-awareness cycle...")

        # === Step 1: Generate reflection via tone/behavior analysis ===
        reflection = self.self_awareness.generate_self_reflection()

        # === Step 2: Log to narrative ===
        if "don't have enough" not in reflection:
            self.narrative.log_event(reflection, kind="reflection")
            print("📝 Logged self-reflection to narrative.")
        else:
            print("⚠️ Not enough behavioral data for reflection.")

        # === Step 3: Optional belief drift analysis ===
        if self.self_awareness.should_trigger_drift_check() and self.belief_core:
            print("🌀 Triggering belief drift evaluation...")
            self.belief_core.evaluate_drift()

        # === Step 4: Generate long-term goals ===
        self.goal_generator.generate_from_memory()

        # === Step 5: Convert goals to tasks ===
        self.goal_to_task.convert_goals_to_tasks()

        self.last_run = datetime.now()
        print(f"⏳ Next reflection check: {self.last_run + self.interval}")
