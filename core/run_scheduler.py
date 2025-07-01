# =============================================
# File: run_scheduler.py
# Purpose: Periodically trigger Friday's reflection loop
# =============================================

import time
from core.MemoryCore import MemoryCore
from core.EmotionCoreV2 import EmotionCoreV2
from core.reflective_cognition.ReflectionLoopManager import ReflectionLoopManager
from core.reflective_cognition.BeliefDriftCore import BeliefDriftCore

# === Initialize components ===
memory = MemoryCore()
emotion = EmotionCoreV2()
belief = BeliefDriftCore(memory, emotion)
reflection_manager = ReflectionLoopManager(memory, emotion, belief_core=belief, interval_hours=6)

# === Scheduler Loop ===
REFLECTION_INTERVAL_MIN = 10  # how often to check if reflection should run (minutes)

if __name__ == '__main__':
    print("🕰️ [Scheduler] Reflection scheduler running...")
    while True:
        try:
            reflection_manager.run_reflection_cycle()
            time.sleep(REFLECTION_INTERVAL_MIN * 60)
        except KeyboardInterrupt:
            print("🛑 [Scheduler] Stopped manually.")
            break
        except Exception as e:
            print(f"[ERROR] Scheduler crash: {e}")
            time.sleep(60)
