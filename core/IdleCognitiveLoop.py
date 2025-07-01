# ============================================
# File: IdleCognitiveLoop.py
# Purpose: Trigger identity reflection when system is idle
# ============================================

import time
from datetime import datetime
#from NarrativeMemoryFusion import NarrativeMemoryFusion
from core.NarrativeMemoryFusion import NarrativeMemoryFusion
from EmotionCore import EmotionCore
from SelfNarrativeCore import SelfNarrativeCore

class IdleCognitiveLoop:
    def __init__(self, memory, identity, emotion_core):
        self.memory = memory
        self.identity = identity
        self.emotion_core = emotion_core
        self.fusion = NarrativeMemoryFusion()

    def run(self, interval_minutes=20):
        while True:
            try:
                # === DREAM REFLECTION ===
                reflection = self.fusion.simulate_internal_event("idle_dream")
                emotion = self.emotion_core.analyze_emotion(reflection)
                self.identity.log_event(reflection, kind="dream")
                self.identity.update_mood(emotion.get("primary", "neutral"))
                print(f"\n💤 Dream Reflection: {reflection.strip()}")

                # === DAILY CHECK-IN INJECTION ===
                from pregnancy_support.core.PregnancySupportCore import PregnancySupportCore
                support = PregnancySupportCore()
                question = support.daily_checkin()
                print(f"\n[CHECK-IN] 🩺 {question}")

            except Exception as e:
                print(f"[IdleLoop] Failed: {e}")
            time.sleep(interval_minutes * 60)

