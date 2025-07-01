# ================================================
# CognitivePipeline.py - With Empathy Fallback Layer
# Purpose: Ensure emotionally safe replies in sensitive domains
# ================================================
import os
import json
import random

class CognitivePipeline:
    def __init__(self, llm_core, emotion_core, vector_memory_core, self_narrative_core, memory_core):
        self.llm = llm_core
        self.emotion = emotion_core
        self.vector_memory = vector_memory_core
        self.narrative = self_narrative_core
        self.memory = memory_core
        # Load soft fallback empathy replies
        self.empathy_replies = self._load_empathy_fallback()

    def _load_empathy_fallback(self):
        try:
            base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pregnancy_support", "empathy"))
            path = os.path.join(base, "soft_replies.json")
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            #print(f"[EMPATHY] Failed to load fallback replies: {e}")
            # Hardcoded fallback replies as a backup
            return [
                "You're not alone.",
                "It's okay to feel this way.",
                "I'm here for you.",
                "You matter, even when it doesn't feel like it."
            ]

    def generate_response(self, user_input):
        #print(f"[Pipeline] Input: {user_input}")
        
        result = self.emotion.analyze_emotion(user_input)
        mood_label = result.get("top_emotion", "neutral")
        #print(f"[Pipeline] Detected mood: {mood_label}")
        
        vector_hits = self.vector_memory.query(user_input, top_k=2)
        #print(f"[Pipeline] Memory hits: {vector_hits}")
        
        memory_context = "\n".join([v["text"] for v in vector_hits])
        
        # Get LLM response (now returns string directly)
        raw = self.llm.prompt(user_input)
        #print(f"[Pipeline] LLM raw response: {raw}")
        
        # Handle empathy fallback
        fallback = random.choice(self.empathy_replies) if mood_label in ["anxious", "sad"] else ""
        
        raw_text = raw.get('reply', '') if isinstance(raw, dict) else str(raw)
        if fallback and not any(phrase in raw_text.lower() for phrase in ["you're not alone", "it's okay", "you matter", "i'm here"]):           #print("[Pipeline] Applied empathy fallback.")        
            # Log events
            self.narrative.log_event(user_input, kind="event", source="user")
            self.narrative.log_event(f"[Mood] {mood_label}", kind="emotion", source="emotion_core")
            self.narrative.log_event(raw, kind="event", source="friday")
            
        # Return clean response text only
        if isinstance(raw, dict):
            return raw.get('reply', str(raw))
        else:
            return str(raw)