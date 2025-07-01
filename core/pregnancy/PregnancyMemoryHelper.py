# ==============================================
# File: core/pregnancy/PregnancyMemoryHelper.py
# Purpose: Log structured pregnancy-related memories
# ==============================================

from datetime import datetime

class PregnancyMemoryHelper:
    def __init__(self, memory_core, identity_core):
        self.memory = memory_core
        self.identity = identity_core

    def log_event(self, text: str, emotion: str = "neutral", trimester: str = None):
        data = {
            "type": "pregnancy_log",
            "text": text,
            "emotion": emotion,
            "trimester": trimester,
            "timestamp": datetime.now().isoformat()
        }
        self.memory.save_memory(data)
        self.memory.save_memory({
            "type": "emotion_trace",
            "emotion": emotion,
            "timestamp": datetime.utcnow().isoformat()
        })

        self.identity.log_event(text, mood=emotion, source="pregnancy")
        return data

    def recall_recent_emotions(self, limit=5):
        """
        Retrieve the most recent logged pregnancy emotions.
        """
        logs = self.memory.get_memory(filter={"type": "pregnancy_log"})
        sorted_logs = sorted(logs, key=lambda x: x.get("timestamp", ""), reverse=True)
        return [(log["emotion"], log["text"]) for log in sorted_logs[:limit]]


# ==========================
# CLI Test Mode
# ==========================
if __name__ == "__main__":
    from core.MemoryCore import MemoryCore
    from core.SelfNarrativeCore import SelfNarrativeCore

    memory = MemoryCore()
    identity = SelfNarrativeCore()
    logger = PregnancyMemoryHelper(memory, identity)

    print("🧾 Pregnancy Memory Helper")
    while True:
        note = input("Log entry (or 'exit'):\n> ").strip()
        if note.lower() == "exit":
            break
        emo = input("Emotion:\n> ").strip()
        tri = input("Trimester:\n> ").strip()
        logged = logger.log_event(note, emo, tri)
        print("✅ Logged:", logged)
