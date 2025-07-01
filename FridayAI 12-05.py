# =====================================
# FridayAI.py – Modular AI Bootloader (Final Working Version)
# =====================================

import os
import sys
import logging
from datetime import datetime
from typing import Dict
from dotenv import load_dotenv

# Add current directory to system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

# Core Modules
from MemoryCore import MemoryCore
from EmotionCore import EmotionCore
from GraphBrainCore import GraphBrainCore
from AutoLearningCore import AutoLearningCore
from SelfQueryingCore import SelfQueryingCore
from ConversationMemory import ConversationMemory
from FuzzyMemorySearch import FuzzyMemorySearch
from GraphReasoner import GraphReasoner
from DialogueCore import DialogueCore
from MemoryReflectionEngine import MemoryReflectionEngine
from MemoryContextInjector import MemoryContextInjector
from LongTermMemory import LongTermMemory
from ContextReasoner import ContextReasoner
from EmotionalJournal import EmotionalJournal
from PlanningCore import PlanningCore
from QueryMemoryCore import QueryMemoryCore
from IntentRouter import IntentRouter
from KnowledgeRouter import KnowledgeRouter
from TransportCore import TransportCore
from GenerativeResponder import generate_response

# Display OpenAI SDK version
try:
    from openai import OpenAI
    client = OpenAI()
    print("🧠 OpenAI SDK (v1.x) loaded successfully")
except ImportError:
    print("⚠️ OpenAI SDK not found. GenerativeResponder may fail.")

# =====================================
# FridayAI Core Class Definition
# =====================================
class FridayAI:
    def __init__(self, memory: MemoryCore, emotion_core: EmotionCore, reasoner: ContextReasoner = None):
        self.memory = memory
        self.emotion_core = emotion_core
        self.context_reasoner = reasoner

        self.detector = IntentRouter(self.memory, self.emotion_core)

        self._configure_logging()
        self._init_components()
        self._init_knowledge_systems()
        self.query_engine = QueryMemoryCore(self.memory)
        self.auto_learner = AutoLearningCore(self.memory, self.graph)

    def _configure_logging(self):
        self.logger = logging.getLogger("FridayAI")
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler('friday_activity.log')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        if not self.logger.handlers:
            self.logger.addHandler(handler)

    def _init_components(self):
        self.graph = GraphBrainCore(self.memory)
        self.auto_learner = AutoLearningCore(self.memory, self.graph)
        self.self_query = SelfQueryingCore(self.memory)
        self.conversation = ConversationMemory()
        self.fuzzy_search = FuzzyMemorySearch(self.memory)
        self.reasoner = GraphReasoner(self.graph)
        self.dialogue = DialogueCore(self.memory, self.reasoner)
        self.reflection = MemoryReflectionEngine(self.memory)
        self.context_injector = MemoryContextInjector()
        self.long_term = LongTermMemory()
        self.context_reasoner = ContextReasoner(self.long_term, self.emotion_core)
        self.journal = EmotionalJournal()
        self.planner = PlanningCore(self.memory)

    def _init_knowledge_systems(self):
        self.domain_handlers = {
            'transport': TransportCore()
        }
        self.router = KnowledgeRouter()
        
    def reflect_state(self, days: int = 7) -> str:
        """
        Uses MemoryReflectionEngine and EmotionalJournal to generate a human-style self-reflection.
        """
        try:
            emotion_trend = self.journal.summarize_range(days)
            thoughts = self.reflection.generate_reflection(emotion_trend)
            return thoughts
        except Exception as e:
            self.logger.error(f"Reflection failed: {str(e)}")
            return "I couldn't generate a reflection right now."

    def respond_to(self, user_input: str) -> Dict[str, object]:
        enriched_input = self.context_injector.inject_context(user_input)
        
        if any(kw in enriched_input.lower() for kw in ["what do you remember about", "recall", "do you remember", "have i mentioned"]):
            results = self.query_engine.query_memory(enriched_input, days=90)

            # 🧠 Filter memory: keep only important entries or ones with specific tags
            filtered = []
            for r in results:
                meta = r.get("metadata", {})
                importance = meta.get("importance", 0)
                tags = meta.get("tags", [])
                source = r.get("source", "")

                if (
                    importance >= 0.5 or
                    source in ["cli", "user", "auto_learned"] or
                    any(tag in tags for tag in ["emotion", "project", "health", "fridayai"])
                ):
                    filtered.append(r)

            if filtered:
                out = "\n".join(f"- [{r['timestamp'].split('T')[0]}] {r.get('value', '')}" for r in filtered)
                return {
                    'domain': 'memory_query',
                    'content': f"🧠 Here's what I remember:\n{out}",
                    'confidence': 0.95,
                    'processing_time': datetime.now().isoformat()
                }
            else:
                return {
                    'domain': 'memory_query',
                    'content': "I couldn’t find anything meaningful in memory related to that.",
                    'confidence': 0.5,
                    'processing_time': datetime.now().isoformat()
                }

        if any(x in enriched_input.lower() for x in ["how have i been feeling", "my mood", "emotions this week", "reflect on my state", "how have i been lately"]):
            history = self.journal.summarize_range(7)
            reflection = self.reflect_state(7)

            if not history:
                return {
                    "domain": "journal",
                    "content": "I haven't tracked enough emotional data yet.",
                    "confidence": 0.5
                }

            lines = [f"{entry['date']}: {entry['dominant']} ({entry['emotions']})" for entry in history]

            return {
                'domain': 'emotion_journal',
                'content': (
                    "Here's your emotional trend this week:\n"
                    + "\n".join(lines)
                    + "\n\nAnd this is how I reflect on your state:\n"
                    + reflection
                ),
                'confidence': 0.95,
                'emotional_tone': history[-1]['dominant'],
                'processing_time': datetime.now().isoformat()
            }

        try:
            context = self.context_reasoner.get_recent_context()
            emotion_summary = context.get("summary", "neutral")
            snippets = context.get("context", {}).get("snippets", [])

            reply = generate_response(enriched_input, emotion_summary, snippets)

            # ✅ Save both user input and Friday's reply into memory
            self.auto_learner.learn_from_input_output(
                user_input=enriched_input,
                ai_output=reply,
                metadata={
                    "emotion": emotion_summary,
                    "timestamp": datetime.now().isoformat()
                }
            )

            return {
                'domain': 'generative',
                'content': reply,
                'confidence': 0.95,
                'emotional_tone': emotion_summary,
                'processing_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            return self._handle_error(e)

    def _handle_error(self, e: Exception) -> Dict:
        import traceback
        print("====== ERROR ======")
        traceback.print_exc()
        print("===================")
        return {
            'domain': 'error',
            'content': "Something went wrong inside FridayAI.",
            'confidence': 0.0,
            'error': str(e)
        }

# ===============================
# CLI Boot Loop for Manual Use
# ===============================
if __name__ == "__main__":
    print("🚀 FridayAI System Initializing...")

    try:
        memory = MemoryCore(memory_file='friday_memory.enc', key_file='memory.key')
        emotion = EmotionCore()
        ai = FridayAI(memory, emotion)

        print("\n" + "=" * 40)
        print("FridayAI Operational – Type 'exit' to quit")
        print("=" * 40 + "\n")

        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ['exit', 'quit']:
                    break
                response = ai.respond_to(user_input)
                print(f"Friday: {response['content']}")
                print("-" * 40)

            except KeyboardInterrupt:
                print("\n🚩 Manual Interrupt Detected – Shutting Down")
                break

    except Exception as e:
        print(f"\n❌ Critical Failure: {e}")
        sys.exit(1)

    finally:
        print("\n🔋 System Safely Powered Down")
