# ==============================================
# File: EngineeringSupportCore.py
# Purpose: Cognitive support for engineering/logistics/diagnostics
# ==============================================

from datetime import datetime

class EngineeringSupportCore:
    def __init__(self, memory, identity, emotion_core):
        self.memory = memory
        self.identity = identity
        self.emotion_core = emotion_core

    def diagnose_issue(self, description: str):
        self.identity.log_event(f"Engineering issue: {description}", kind="diagnostic")
        keywords = self._extract_keywords(description)

        self.memory.store("engineering_diagnosis", {
            "description": description,
            "keywords": keywords,
            "timestamp": datetime.now().isoformat()
        })

        return f"I’ve logged the issue. Are you looking to repair, replace, or analyze further?"

    def _extract_keywords(self, text: str):
        # Very simple keyword extractor (can be upgraded later)
        return [word for word in text.lower().split() if len(word) > 4]

    def suggest_next_step(self, fault_type: str):
        if "hydraulic" in fault_type:
            return "Check pressure seals and actuator leaks. Do you need part specs?"
        if "electrical" in fault_type:
            return "Check continuity and circuit integrity first. Want a voltmeter protocol?"
        return "Let's isolate the system first — mechanical, electrical, or sensor issue?"

    def query_memory_by_part(self, part_name: str):
        logs = self.memory.search("engineering_diagnosis")
        relevant = [entry for entry in logs if part_name.lower() in entry["description"].lower()]
        return relevant or f"No known issues logged with part: {part_name}"
