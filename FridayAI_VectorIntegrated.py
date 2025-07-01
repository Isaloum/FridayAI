
import os
import logging
import sys
from datetime import datetime
from typing import Dict, Optional

from MemoryCore import MemoryCore
from AutoLearningCore import AutoLearningCore
from EmotionCore import EmotionCore
from TransportCore import TransportCore
from KnowledgeRouter import KnowledgeRouter
from SelfQueryingCore import SelfQueryingCore
from VectorMemoryCore import VectorMemoryCore
from dotenv import load_dotenv

load_dotenv()

class FridayAI:
    """FridayAI: A modular, emotionally aware cognitive engine with semantic recall"""

    def __init__(self):
        self._configure_logging()
        self._init_components()
        self._init_knowledge_systems()

    def _configure_logging(self):
        self.logger = logging.getLogger("FridayAI")
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler('friday_activity.log')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(module)s - %(message)s'
        ))
        if not self.logger.handlers:
            self.logger.addHandler(handler)

    def _init_components(self):
        self.memory = MemoryCore()
        self.emotion = EmotionCore()
        self.auto_learner = AutoLearningCore(self.memory)
        self.self_query = SelfQueryingCore(self.memory)
        self.vector_memory = VectorMemoryCore()
        self.router = KnowledgeRouter()

    def _init_knowledge_systems(self):
        self.domain_handlers = {
            'transport': TransportCore()
        }

    def respond_to(self, user_input: str, session_id: str = "default-session") -> Dict[str, object]:
        try:
            emotion = self.emotion.analyze(user_input)

            # Step 1: Check learned memory
            if memory_response := self._check_learned_facts(user_input):
                return self._enhance_response(memory_response, user_input, emotion)

            # Step 2: Domain handling
            domain = self.router.detect_domain(user_input)
            if domain and domain in self.domain_handlers:
                response = self.domain_handlers[domain].handle_query(user_input)
                return self._enhance_response(response, user_input, emotion)

            # Step 3: Router semantic/logic fallback
            response = self.router.handle_query(user_input)
            if response["confidence"] >= 0.5:
                return self._enhance_response(response, user_input, emotion)

            # Step 4: Vector memory semantic search
            matches = self.vector_memory.query(user_input, top_k=1)
            if matches:
                best = matches[0]
                return self._enhance_response({
                    "domain": "vector_memory",
                    "content": f"I've seen something like this before:
"{best['text']}"
(Similarity: {round(best['score'], 2)})",
                    "confidence": best['score']
                }, user_input, emotion)

            # Step 5: Auto-learning fallback
            if self.auto_learner.process(user_input):
                followups = self.self_query.suggest_followups(user_input)
                return self._enhance_response({
                    'domain': 'learning',
                    'content': "I've updated my knowledge base.\n\nFollow-up ideas:\n" + "\n".join(followups),
                    'confidence': 0.85
                }, user_input, emotion)

            return self._enhance_response({
                'domain': 'general',
                'content': "I'm still learning about that. Could you explain more?",
                'confidence': 0.0
            }, user_input, emotion)

        except Exception as e:
            import traceback
            print("====== DEBUG ERROR ======")
            print(str(e))
            traceback.print_exc()
            print("====== END DEBUG ======")
            return {
                'status': 'error',
                'content': "System temporarily unavailable",
                'error_code': 500
            }

    def _enhance_response(self, response: Dict, query: str, emotion: str) -> Dict:
        tone_msg = self.emotion.get_emotional_response(emotion)
        return {
            **response,
            'emotional_tone': emotion,
            'processing_time': datetime.now().isoformat(),
            'query_type': response.get('domain', 'unknown'),
            'content': f"{tone_msg}\n\n{response['content']}"
        }

    def _check_learned_facts(self, query: str) -> Optional[Dict]:
        learned_keys = ['identity/name', 'geo/location', 'bio/birthdate']
        for key in learned_keys:
            if key in query.lower() and (value := self.memory.get_fact(key)):
                return {
                    'domain': 'memory',
                    'content': f"Your {key.split('/')[-1]} is {value}",
                    'confidence': 1.0
                }
        return None

if __name__ == "__main__":
    print("🚀 FridayAI System Initializing...")
    sys.stdout.flush()

    try:
        ai = FridayAI()
        print("\n" + "="*40)
        print("FridayAI Operational - Type 'exit' to quit")
        print("="*40 + "\n")

        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ('exit', 'quit'):
                    break
                response = ai.respond_to(user_input, session_id="user-001")
                print(f"\nFriday: {response['content']}")
                print("-"*40)
            except KeyboardInterrupt:
                print("\n🛑 Emergency shutdown initiated!")
                break

    except Exception as e:
        print(f"\n⚠️ Critical Failure: {str(e)}")
        sys.exit(1)
    finally:
        print("\n🔋 System Safely Powered Down")
        sys.stdout.flush()
