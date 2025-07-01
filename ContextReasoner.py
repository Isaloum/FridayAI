# =========================================
# ContextReasoner.py – Cognitive Context Analysis for FridayAI
# =========================================

from datetime import datetime
import random
import re

class ContextReasoner:
    def __init__(self, memory_core, emotion_core):
        self.memory = memory_core
        self.emotion_core = emotion_core

    def analyze(self, user_input: str) -> dict:
        """
        Analyzes the user's input to infer high-level context traits without keywords.
        Returns trajectory, volition, urgency, and cognitive mode.
        """
        now = datetime.now()
        mem_recent = self.memory.fetch_recent(limit=5, within_minutes=90)
        emotion_scan = self.emotion_core.analyze(user_input)

        # === Trajectory Inference ===
        trajectory = self._estimate_trajectory(user_input, mem_recent)

        # === Volition (Willpower / Clarity) ===
        volition_strength = self._estimate_volition(user_input)

        # === Urgency Level ===
        urgency_score = self._estimate_urgency(user_input)

        # === Cognitive Intent Mode ===
        intent_type = self._infer_intent_type(user_input, emotion_scan)

        # === Meta Flags (fractured identity, trauma loops, AI awareness, etc.) ===
        meta = self._detect_meta_state(user_input)

        return {
            "trajectory": trajectory,
            "volition_strength": round(volition_strength, 2),
            "urgency_score": round(urgency_score, 2),
            "intent_type": intent_type,
            "meta": meta,
            "timestamp": now.isoformat()
        }

    def _estimate_trajectory(self, text, memory):
        sentiment_shifts = [self.emotion_core.analyze(m["content"]).get("dominant", "") for m in memory]
        if len(set(sentiment_shifts)) <= 2:
            return "loop"
        elif any(w in text.lower() for w in ["future", "next", "tomorrow"]):
            return "forward"
        else:
            return "stuck"

    def _estimate_volition(self, text):
        tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text)
        strength = len([w for w in tokens if w.lower() in ["start", "build", "fix", "plan", "reset", "change", "own"]])
        return min(1.0, 0.2 + strength * 0.15)

    def _estimate_urgency(self, text):
        text = text.lower()
        high = ["now", "need", "urgent", "can’t wait", "help"]
        urgency = sum(text.count(h) for h in high)
        return min(1.0, urgency * 0.2)

    def _infer_intent_type(self, text, emotion_map):
        score = emotion_map.get("confused", 0) + emotion_map.get("hopeful", 0) + emotion_map.get("curious", 0)
        if score > 0.9:
            return "self_optimizing"
        elif emotion_map.get("uncertain", 0) > 0.5:
            return "truth_check"
        elif emotion_map.get("overwhelmed", 0) > 0.5:
            return "emotional_dump"
        elif len(text) > 80 and emotion_map.get("neutral", 0) > 0.4:
            return "observation_log"
        elif emotion_map.get("pain", 0) > 0.3:
            return "body_check"
        elif emotion_map.get("curious", 0) > 0.4:
            return "learn_mode"
        else:
            return "undefined"

    def _detect_meta_state(self, text):
        if "who am i" in text.lower() or "what’s wrong with me" in text.lower():
            return "identity_frag"
        return "none"

    def get_recent_context(self, limit=3, timeframe_minutes=60) -> dict:
        return self.memory.fetch_recent(limit=limit, within_minutes=timeframe_minutes)
