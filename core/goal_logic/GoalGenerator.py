# =============================================
# File: GoalGenerator.py (Option A)
# Purpose: Generate long-term goals from recent memory + emotion
# =============================================

import json
from core.MemoryCore import MemoryCore
from core.EmotionCoreV2 import EmotionCoreV2
from core.goal_logic.GoalManager import GoalManager
from datetime import datetime

class GoalGenerator:
    def __init__(self):
        self.memory = MemoryCore()
        self.emotion = EmotionCoreV2()
        self.manager = GoalManager()

    def generate_from_memory(self):
        recent = self.memory.query_memories(since_hours=72)
        print(f"[DEBUG] Found {len(recent)} recent memory entries")
        for r in recent:
            print(json.dumps(r, indent=2))

        emotion = self.emotion.mood

        candidates = []
        for mem in recent:
            print("\n[DEBUG] Memory entry being evaluated:")
            print(json.dumps(mem, indent=2))  # Requires import json at top

            content = mem.get("content", "")
            tags = mem.get("tags", [])
            mem_type = mem.get("type", "")

            if (
                mem_type == "task_execution" and
                any(tag in ["urgent", "risk", "danger", "health", "pregnancy_docs"] for tag in tags)
            ):
                print("[DEBUG] → Match found. Will create goal.")
                candidates.append({
                    "description": f"Follow-up on: {content[:80]}...",
                    "emotion": mem.get("emotion", "neutral"),
                    "tags": tags,
                    "converted_to_task": False,
                    "status": "active",
                    "origin": "memory_scan"
                })
            else:
                print("[DEBUG] → Skipped. Either type or tags did not match.")


        for item in candidates:
            self.manager.add_goal(
                description=item["description"],
                tags=item["tags"],
                emotion=item["emotion"],
                origin=item["origin"],
                extra_fields={
                    "converted_to_task": item["converted_to_task"],
                    "status": item["status"]
                }
            )

        print(f"[GoalGenerator] Proposed {len(candidates)} new goal(s).")
        return candidates


if __name__ == '__main__':
    g = GoalGenerator()
    g.generate_from_memory()
