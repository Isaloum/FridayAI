# ReflectionScheduler.py
# --------------------------
# Periodically summarizes long-term memory and stores reflective insights.

from datetime import datetime, timedelta
from MemoryReflectionEngine import MemoryReflectionEngine
from LongTermMemory import LongTermMemory
from MemoryCore import MemoryCore

class ReflectionScheduler:
    def __init__(self, interval_minutes: int = 1440):  # Default: once per day
        self.memory_core = MemoryCore()
        self.long_term = LongTermMemory()
        self.engine = MemoryReflectionEngine(self.memory_core)
        self.last_run = None
        self.interval = timedelta(minutes=interval_minutes)

    def should_run(self):
        now = datetime.now()
        if self.last_run is None:
            return True
        return (now - self.last_run) >= self.interval

    def run_reflection(self):
        if not self.should_run():
            return None

        recent = self.long_term.fetch_recent(limit=12, within_minutes=1440)
        if not recent:
            return None

        notes = []
        for entry in recent:
            thought = self.engine.reflect_on(entry["user_input"], entry["reply"])
            if thought:
                self.memory_core.save_fact("reflection.note", thought, source="auto_reflection")
                notes.append(thought)

        self.last_run = datetime.now()
        return notes


# Manual test
if __name__ == "__main__":
    scheduler = ReflectionScheduler()
    thoughts = scheduler.run_reflection()
    if thoughts:
        print("🧠 Reflections saved:")
        for t in thoughts:
            print("-", t)
    else:
        print("(No reflections needed or nothing new to process.)")
