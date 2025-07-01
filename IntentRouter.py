# IntentRouter.py – Now with response content for semantic intents

from datetime import datetime
from typing import Dict
from EmotionCore import EmotionCore
from ContextReasoner import ContextReasoner
from MemoryCore import MemoryCore
from IntentDetector import IntentDetector
import json
import os

MISROUTE_LOG = "friday_misroutes.log"
DEBUG_MODE = os.getenv("FRIDAY_DEBUG", "false").lower() == "true"

class IntentRouter:
    def __init__(self, memory_core: MemoryCore, emotion_core: EmotionCore, context_reasoner: ContextReasoner):
        self.memory = memory_core
        self.emotion_core = emotion_core
        self.context_reasoner = context_reasoner
        self.intent_model = IntentDetector()
        self.traits = {
            "warmth": 0.5,
            "humor": 0.5,
            "formality": 0.5,
            "precision": 0.5
        }

    def normalize_input(self, text: str) -> str:
        import re
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
        text = re.sub(r"\s+", " ", text).strip()
        slang_map = {
            "u": "you",
            "r": "are",
            "yo": "hey",
            "gurl": "girl",
            "wanna": "want to",
            "gonna": "going to",
            "luv": "love"
        }
        return " ".join(slang_map.get(word, word) for word in text.split())

    def save_traits(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(self.traits, f)

    def load_traits(self, filepath: str):
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                self.traits = json.load(f)

    def route(self, user_input: str) -> Dict[str, object]:
        user_input = self.normalize_input(user_input)
        if DEBUG_MODE:
            print(f"🧠 User Input: {user_input}")
        timestamp = datetime.now().isoformat()
        context = self.context_reasoner.analyze(user_input)
        emotion = self.emotion_core.analyze(user_input)
        vector = self.intent_model.vector_profile(user_input)
        if DEBUG_MODE:
            print(f"🔍 Intent Vector Profile: {vector}")

        volition = context.get("volition_strength", 0.0)
        urgency = context.get("urgency_score", 0.0)
        trajectory = context.get("trajectory", "neutral")
        intent_type = context.get("intent_type", "undefined")
        tone = emotion.get("dominant", "neutral")

        scores = {
            "review_missions": volition * 0.4 + vector.get("goal_oriented", 0.3) * 0.3 + (1.0 if trajectory == "forward" else 0) * 0.3,
            "reflect_goals": vector.get("progress_reflection", 0.3) * 0.5 + volition * 0.2 + (1.0 if trajectory == "loop" else 0) * 0.3,
            "belief_drift_check": vector.get("identity_check", 0.3) * 0.5 + emotion.get("uncertain", 0.2) * 0.3 + (1.0 if intent_type == "truth_check" else 0) * 0.2,
            "vent": emotion.get("overwhelmed", 0.4) * 0.5 + urgency * 0.4 + vector.get("emotional_dump", 0.2) * 0.1,
            "journal": urgency * -1.0 + volition * 0.2 + vector.get("observation_log", 0.3),
            "log_symptom": emotion.get("pain", 0.4) * 0.4 + urgency * 0.3 + vector.get("health_monitoring", 0.3),
            "learn": vector.get("curiosity", 0.5) * 0.4 + volition * 0.3 + urgency * 0.2,
            "default_chat": vector.get("reflect", 0.4) * 0.5 + urgency * 0.1 + volition * 0.1
        }

        if vector.get("health_monitoring", 0) < 0.2 and emotion.get("pain", 0) < 0.2:
            if DEBUG_MODE:
                print("🛑 log_symptom suppressed due to weak signal")
            scores["log_symptom"] = 0.0

        sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if DEBUG_MODE:
            print(f"🏆 Intent Scores: {scores}")
        best_intent, best_score = sorted_intents[0]

        # === [MISROUTE FALLBACK] ===
        if best_score < 0.25:
            original = best_intent
            best_intent = "default_chat"

            with open("friday_misroutes.log", "a") as log:
                log.write(f"[{timestamp}] Fallback from '{original}' → 'default_chat'\n")
                log.write(f"  Input: {user_input}\n")
                log.write(f"  Emotion: {tone} | Vector: {vector}\n\n")

            if DEBUG_MODE:
                print(f"⚠️ Low confidence ({best_score:.2f}) — routed to 'default_chat' instead of '{original}'")

        return {
            "domain": best_intent,
            "confidence": best_score,
            "emotional_tone": tone,
            "timestamp": timestamp
        }

