# =============================================
# File: BeliefDriftCore.py
# Purpose: Detects belief drift or internal contradictions
# =============================================

import json
from datetime import datetime, timedelta
from pathlib import Path

BELIEF_LOG_PATH = Path("core/memory_bank/belief_log.json")
MEMORY_LOG_PATH = Path("core/memory_bank/memory_log.json")

class BeliefDriftCore:
    def __init__(self):
        self.belief_path = BELIEF_LOG_PATH
        self.memory_path = MEMORY_LOG_PATH
        self.beliefs = self._load_beliefs()
        self.memory = self._load_memory()

    def _load_beliefs(self):
        if not self.belief_path.exists():
            return []
        with open(self.belief_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_memory(self):
        if not self.memory_path.exists():
            return []
        with open(self.memory_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def evaluate_drift(self):
        """
        Look at recent memory entries and check for tone or topic contradiction
        compared to last beliefs. For now, simple heuristic based on tag mismatch.
        """
        recent = [m for m in self.memory if self._is_recent(m)]

        drift_events = []
        for belief in self.beliefs[-10:]:
            for mem in recent:
                if self._contradicts(belief, mem):
                    drift_events.append({
                        "timestamp": datetime.now().isoformat(),
                        "belief": belief,
                        "conflict": mem,
                        "note": "Possible contradiction or evolution detected."
                    })

        if drift_events:
            self._save_drift(drift_events)
            print(f"[BeliefDrift] ?? Detected {len(drift_events)} drift event(s).")
        else:
            print("[BeliefDrift] No significant belief drift found.")

    def _is_recent(self, mem):
        try:
            ts = datetime.fromisoformat(mem.get("timestamp", datetime.now().isoformat()))
            return ts >= datetime.now() - timedelta(hours=48)
        except Exception:
            return False

    def _contradicts(self, belief, mem):
        # Simple mock logic for contradiction detection
        # Later can be upgraded to use NLP contradiction classifiers
        belief_tags = set(belief.get("tags", []))
        mem_tags = set(mem.get("tags", []))
        return not belief_tags.isdisjoint(mem_tags) and belief.get("tone") != mem.get("tone")

    def _save_drift(self, drift_events):
        self.beliefs.extend(drift_events)
        with open(self.belief_path, 'w', encoding='utf-8') as f:
            json.dump(self.beliefs, f, indent=2)
