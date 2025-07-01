# =============================================
# File: TaskReflectionAgent.py
# Purpose: Reflect on individual task results after execution
# =============================================

import datetime
from core.MemoryCore import MemoryCore
from core.EmotionCoreV2 import EmotionCoreV2
from core.memory_bank.SelfNarrativeCore import SelfNarrativeCore


class TaskReflectionAgent:
    """
    This agent reflects on each task result immediately after execution.
    It summarizes outcomes, adjusts emotional state, and injects insights into memory.
    """

    def __init__(self):
        self.memory_core = MemoryCore()
        self.emotion_core = EmotionCoreV2()
        self.narrative = SelfNarrativeCore()  # ✅ Add this


    def summarize_task_results(self, task_result: dict) -> str:
        task_id = task_result.get("task_id", "unknown")
        desc = task_result.get("description", "No description")
        result = task_result.get("result", "No result")
        timestamp = task_result.get("timestamp", str(datetime.datetime.now()))

        summary = f"Task ID: {task_id}\nDescription: {desc}\nResult: {result}\nCompleted at: {timestamp}"
        return summary

    def inject_reflection_to_memory(self, summary: str, metadata: dict):
        memory_entry = {
            "type": "reflection",
            "content": summary,
            "tags": ["reflection", metadata.get("task_id", "unknown")],
            "timestamp": metadata.get("timestamp", str(datetime.datetime.now())),
            "emotion": metadata.get("emotion", "neutral")
        }
        self.memory_core.save_memory(memory_entry)

    def update_emotional_tone(self, summary: str):
        if "error" in summary.lower() or "failed" in summary.lower():
            self.emotion_core.adjust_mood(-0.1)
        else:
            self.emotion_core.adjust_mood(0.1)

    def run_reflection_cycle(self, task_result: dict):
        summary = self.summarize_task_results(task_result)
        metadata = {
            "task_id": task_result.get("task_id"),
            "timestamp": task_result.get("timestamp"),
            "emotion": task_result.get("emotion", "neutral")
        }
        self.inject_reflection_to_memory(summary, metadata)
        self.narrative.log_event(summary, kind="reflection")
        self.update_emotional_tone(summary)
        print("[TaskReflectionAgent] Reflection complete for task:", task_result.get("task_id"))


# =============================================
# PATCH TO APPLY IN: task_executor.py
# =============================================
#
# from TaskReflectionAgent import TaskReflectionAgent
# reflection_agent = TaskReflectionAgent()
# ... after task execution:
# reflection_agent.run_reflection_cycle(task_result)
# =============================================
