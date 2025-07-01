# ==============================================
# File: FridayCoreEngine.py
# Purpose: Main brain loop that runs Friday as a unified cognitive system
# ==============================================

import time
from datetime import datetime
from FridayAI import FridayAI
from MemoryCore import MemoryCore
from EmotionCore import EmotionCore
from DomainAdapterCore import DomainAdapterCore

# === System Boot ===
print("\n[BOOT] Friday Core Engine Starting Up...")

# === Core Memory and Emotion Systems ===
from EmotionCoreV2 import EmotionCoreV2   # Make sure this line is present and ABOVE the call

emotion = EmotionCoreV2()
memory = MemoryCore(memory_file="friday_memory.enc", key_file="memory.key")

# === Instantiate Friday ===
ai = FridayAI(memory, emotion)


# === Load Default Domain (Can switch later) ===
ai.domain_adapter.load_domain("default", domain_context={})

# === Setup Self-Awareness Reflection Loop ===
def reflection_cycle():
    try:
        ai.reflection_loop.run_reflection_cycle(
            self_awareness=ai.self_awareness,
            belief_updater=ai.belief_updater
        )
    except Exception as e:
        print(f"[ERROR] Reflection cycle failed: {e}")

# === Idle Dream Cycle ===
def dream_cycle():
    try:
        dream = ai.narrative_fusion.simulate_internal_event("idle_dream")
        mood = emotion.analyze_emotion(dream).get("primary", "neutral")
        ai.identity.log_event(dream, kind="dream")
        ai.identity.update_mood(mood)
        print(f"\n[DREAM] {dream.strip()}  \n[MOOD] → {mood}")
    except Exception as e:
        print(f"[ERROR] Dream cycle failed: {e}")

# === Live Cognitive Loop ===
while True:
    try:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break

        if user_input.lower().startswith("load_domain:"):
            domain = user_input.split(":", 1)[1].strip()
            ai.domain_adapter.load_domain(domain, domain_context={})
            continue

        if user_input.lower() == "reflect":
            reflection_cycle()
            continue

        if user_input.lower() == "dream":
            dream_cycle()
            continue

        response = ai.respond_to(user_input)
        print(f"Friday: {response['content']}")

    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"[ERROR] Main loop failure: {e}")

# === Graceful Shutdown ===
print("\n[SHUTDOWN] Friday Core Engine Terminated")
try:
    ai.router.save_traits("traits.json")
except:
    pass
