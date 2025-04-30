"""
FridayAI - Modular Artificial Intelligence Framework
Version: 1.2.0
Author: [Your Name]
License: MIT
"""

# === Standard Library Imports ===
import os
import re
import time
import logging
from datetime import datetime
from typing import Dict, Optional

# === Third-Party Imports ===
import pyttsx3
import requests
from dotenv import load_dotenv

# === Local Core Modules ===
from MemoryCore import MemoryCore
from EmotionCore import EmotionCore
from KnowledgeCore import KnowledgeCore
from AutoLearningCore import AutoLearningCore
from SelfQueryingCore import SelfQueryingCore
from PersonalDataCore import PersonalDataCore

# === Routing for Domains ===
class KnowledgeRouter:
    def __init__(self):
        self.domain_patterns = {
            'transport': [
                r'\b(ride|transport|airport|yul|laval|bus|taxi|metro)\b',
                r'\b(get|go|travel) to\b',
                r'\bhow to (get|reach)\b'
            ],
            'medical': [
                r'\b(pain|hospital|medicine|doctor|symptom)\b',
                r'\bhealth (issue|problem)\b'
            ]
        }

    def detect_domain(self, text: str) -> Optional[str]:
        text = text.lower()
        domain_scores = {}

        for domain, patterns in self.domain_patterns.items():
            match_count = sum(1 for pattern in patterns if re.search(pattern, text))
            domain_scores[domain] = match_count / len(patterns)

        max_domain = max(domain_scores, key=domain_scores.get)
        return max_domain if domain_scores[max_domain] > 0.65 else None


# === Transport Domain Knowledge Handler ===
class TransportCore:
    def __init__(self):
        self.knowledge_base = {
            'yul_transport': {
                'response': (
                    "From Laval to YUL Airport:\n"
                    "1. Taxi/Uber: 40-60$ CAD\n"
                    "2. 747 Bus: Lionel-Groulx metro\n"
                    "3. Airport Shuttle: 1-800-123-4567\n"
                    "4. Car Rental: Available at YUL"
                ),
                'keywords': ['yul', 'airport', 'transport', 'laval']
            }
        }

    def handle_query(self, query: str) -> Dict[str, object]:
        response = {
            'domain': 'transport',
            'confidence': 0.0,
            'content': None,
            'sources': []
        }

        for entry in self.knowledge_base.values():
            keyword_matches = sum(1 for kw in entry['keywords'] if kw in query.lower())
            confidence = keyword_matches / len(entry['keywords'])

            if confidence > response['confidence']:
                response.update({
                    'confidence': confidence,
                    'content': entry['response'],
                    'sources': entry['keywords']
                })

        return response


# === FridayAI Master Brain ===
class FridayAI:
    def __init__(self, memory_core: Optional[MemoryCore] = None):
        self._init_logging()
        self._load_environment()
        self._init_subcomponents(memory_core)
        self._init_speech_engine()

        self.domain_cores = {
            'transport': TransportCore()
        }
        self.knowledge_router = KnowledgeRouter()
        self.api_endpoint = os.getenv("API_ENDPOINT", "https://api.openai.com/v1/chat/completions")

    def _init_logging(self):
        self.logger = logging.getLogger("FridayAI")
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler('friday_operations.log')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s'))
        self.logger.addHandler(handler)

    def _load_environment(self):
        if not load_dotenv():
            self.logger.warning("No .env file found, using system environment")

    def _init_subcomponents(self, memory_core):
        self.memory = memory_core or MemoryCore()
        self.emotion_core = EmotionCore()
        self.auto_learning_core = AutoLearningCore(self.memory)
        self.self_querying_core = SelfQueryingCore(self.memory)
        self.personal_data_core = PersonalDataCore(self.memory)

    def _init_speech_engine(self):
        try:
            self.speech_engine = pyttsx3.init()
            self.speech_engine.setProperty('rate', 150)
            self.speech_engine.setProperty('voice', 'english')
        except RuntimeError as e:
            self.logger.error(f"Speech engine initialization failed: {str(e)}")
            self.speech_engine = None

    def process_query(self, user_input: str) -> Dict[str, object]:
        try:
            detected_domain = self.knowledge_router.detect_domain(user_input)
            if detected_domain in self.domain_cores:
                domain_response = self.domain_cores[detected_domain].handle_query(user_input)
            else:
                domain_response = self._handle_general_query(user_input)

            return self._apply_postprocessing(domain_response, user_input)

        except Exception as e:
            self.logger.error(f"Query processing failed: {str(e)}")
            return {
                'status': 'error',
                'message': 'System processing error',
                'error_code': 500
            }

    def _handle_general_query(self, query: str) -> Dict[str, object]:
        return {
            'domain': 'general',
            'confidence': 0.0,
            'content': "I need more context to answer that.",
            'sources': []
        }

    def _apply_postprocessing(self, response: Dict, original_query: str) -> Dict:
        enhanced = {
            'metadata': {
                'processing_steps': ['domain_detection', 'knowledge_retrieval'],
                'temporal_context': {
                    'received_at': datetime.now().isoformat(),
                    'processing_time': None
                }
            },
            'original_query': original_query,
            'domain': response['domain'],
            'confidence': round(response['confidence'], 2),
            'content': response['content'],
            'sources': response['sources']
        }
        enhanced['emotional_context'] = self.emotion_core.analyze(original_query)
        return enhanced

    def vocalize_response(self, text: str):
        if self.speech_engine:
            try:
                self.speech_engine.say(text)
                self.speech_engine.runAndWait()
            except Exception as e:
                self.logger.error(f"Speech synthesis failed: {str(e)}")

# === Test Block ===
if __name__ == "__main__":
    ai = FridayAI()
    query = "How do I get to the airport from Laval?"
    result = ai.process_query(query)
    print(f"Response: {result['content']}")
