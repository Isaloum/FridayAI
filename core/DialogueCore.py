# =====================================
# DialogueCore.py – Contextual Dialogue Engine with Graph Reasoning
# =====================================

from datetime import datetime
from typing import Dict
import re

from QueryMemoryCore import QueryMemoryCore
from GraphBrainCore import GraphBrainCore
from ReflectionCore import ReflectionCore
from core.PlanningCore import PlanningCore

class DialogueCore:
    """Handles natural conversation by combining memory, emotion, and logic."""

    def __init__(self, memory_core, emotion_core):
        self.memory = memory_core
        self.emotion = emotion_core
        self.query_engine = QueryMemoryCore(memory_core)
        self.brain = GraphBrainCore(memory_core)
        self.reflector = ReflectionCore(memory_core)
        self.planner = PlanningCore(memory_core)

    def respond_to(self, user_input: str) -> Dict:
        """
        Generates a context-aware response based on memory, emotion, and conceptual linking.
        """
        user_input_lower = user_input.lower()

        # ----------------------------------------
        # MEMORY SUMMARIZATION TRIGGER
        # ----------------------------------------
        if re.search(r"(what.*remember|summarize.*week|talked.*about|recall)", user_input_lower):
            summary = self.query_engine.summarize()
            return {
                "domain": "memory_summary",
                "content": summary,
                "confidence": 0.95,
                "timestamp": datetime.now().isoformat()
            }

        # ----------------------------------------
        # SPECIFIC TAG-BASED MEMORY QUERIES
        # ----------------------------------------
        if re.search(r"(remember.*health|pain|accident|injury|project)", user_input_lower):
            if "health" in user_input_lower:
                summary = self.query_engine.summarize(tag="health")
            elif "pain" in user_input_lower:
                summary = self.query_engine.summarize(tag="pain")
            elif "project" in user_input_lower:
                summary = self.query_engine.summarize(tag="project")
            elif "injury" in user_input_lower or "accident" in user_input_lower:
                summary = self.query_engine.summarize(tag="injury")
            else:
                summary = self.query_engine.summarize()

            return {
                "domain": "tag_summary",
                "content": summary,
                "confidence": 0.95,
                "timestamp": datetime.now().isoformat()
            }

        # ----------------------------------------
        # DYNAMIC EMOTIONAL TREND RECALL
        # ----------------------------------------
        emotion_query_patterns = [
            r"\bhow (was|were) i\b",
            r"\bhow.*feel.*(yesterday|last week|last time)\b",
            r"\bwhat.*emotion.*\b",
            r"\bmood.*(lately|last week|yesterday)\b",
            r"\b(feeling|felt).*(recent|last few days|last time)\b",
            r"\bhow.*(have i|i been).*feel.*\b"
        ]

        if any(re.search(p, user_input_lower) for p in emotion_query_patterns):
            if self.emotion:
                trend = self.emotion.get_emotion_trend(days=7)
                if trend["total_mentions"] > 0:
                    return {
                        "domain": "emotion",
                        "content": (
                            f"Here's your emotional trend this week:\n"
                            f"{trend['date']}: {trend['dominant']} ({trend['emotions']})"
                        ),
                        "confidence": 0.95,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    return {
                        "domain": "emotion",
                        "content": "I didn’t detect any emotional entries in the last few days. Want to log how you're feeling now?",
                        "confidence": 0.7,
                        "timestamp": datetime.now().isoformat()
                    }

        # ----------------------------------------
        # FREQUENT TOPIC / TAG QUERY
        # ----------------------------------------
        if re.search(r"(common|most frequent|top tags|topics)", user_input_lower):
            top_tags = self.query_engine.get_frequent_topics()
            return {
                "domain": "topic_stats",
                "content": "Top recurring topics:\n- " + "\n- ".join(top_tags),
                "confidence": 0.9,
                "timestamp": datetime.now().isoformat()
            }

        # ----------------------------------------
        # DYNAMIC ASSOCIATION REFLECTION (GraphBrainCore)
        # ----------------------------------------
        concept_patterns = [
            r"\bconnected to\b", r"\blinked to\b", r"\brelated to\b",
            r"\bassociation with\b", r"\bwhy.*keep mentioning\b",
            r"\bwhat’s around\b", r"\bcomes up with\b"
        ]

        if any(re.search(p, user_input_lower) for p in concept_patterns):
            match = re.findall(r"\b(?:to|with|about|on|of)?\s*(\w+)$", user_input_lower)
            if match:
                topic = match[-1]
                brain_response = self.brain.explain_links_naturally(topic)
                return {
                    "domain": "concept_links",
                    "content": brain_response,
                    "confidence": 0.92,
                    "timestamp": datetime.now().isoformat()
                }
                
        # ----------------------------------------
        # SELF-AWARE REFLECTION PATTERN (ReflectionCore)
        # ----------------------------------------

        reflection_patterns = [
            r"\b(noticed.*about me|pattern.*i keep|repeating.*feeling|something.*keep saying)\b",
            r"\bkeep bringing up\b", r"\bis there something\b.*(i'm missing|i should know)\b",
            r"\banything i should\b.*(notice|be aware|know)\b",
            r"\bwhat.*noticed.*lately\b", r"\bdo you see.*pattern\b"
        ]

        if any(re.search(p, user_input_lower) for p in reflection_patterns):
            insight = self.reflector.reflect_on_patterns(days=7)
            return {
                "domain": "reflection",
                "content": insight,
                "confidence": 0.93,
                "timestamp": datetime.now().isoformat()
            }
            
        # ----------------------------------------
        # PROACTIVE PLANNING INTENT (PlanningCore)
        # ----------------------------------------

        planning_patterns = [
            r"\bhow.*help.*me\b",
            r"\bcan you support\b", r"\bwhat should i do\b",
            r"\bsuggestions?\b", r"\bguide me\b", r"\bwhat.*recommend\b",
            r"\banything i should\b.*(do|change|try)\b"
        ]

        if any(re.search(p, user_input_lower) for p in planning_patterns):
            plan = self.planner.suggest_plan(days=7)
            return {
                "domain": "planning",
                "content": plan,
                "confidence": 0.94,
                "timestamp": datetime.now().isoformat()
            }



        # ----------------------------------------
        # DEFAULT FALLBACK
        # ----------------------------------------
        return {
            "domain": "fallback",
            "content": "I'm still learning. Could you rephrase that or ask about something you've mentioned before?",
            "confidence": 0.5,
            "timestamp": datetime.now().isoformat()
        }
