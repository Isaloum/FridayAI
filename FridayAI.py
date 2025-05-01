# =====================================
# FridayAI.py - Core AI Brain Module
# =====================================
import os
import re
import logging
from datetime import datetime
from typing import Dict, Optional, List
from difflib import get_close_matches

import pyttsx3
import requests
from dotenv import load_dotenv
from cryptography.fernet import Fernet

from MemoryCore import MemoryCore
from EmotionCore import EmotionCore
from KnowledgeCore import KnowledgeCore
from AutoLearningCore import AutoLearningCore
from SelfQueryingCore import SelfQueryingCore
from PersonalDataCore import PersonalDataCore

# ========================
# KNOWLEDGE ROUTING SYSTEM
# ========================
class KnowledgeRouter:
    """Determines which knowledge domain a query belongs to"""
    
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
        """Identify the most relevant knowledge domain"""
        text = text.lower()
        domain_scores = {}

        for domain, patterns in self.domain_patterns.items():
            match_count = sum(1 for pattern in patterns if re.search(pattern, text))
            domain_scores[domain] = match_count / len(patterns)

        max_domain = max(domain_scores, key=domain_scores.get)
        return max_domain if domain_scores[max_domain] > 0.65 else None

# =====================
# TRANSPORT KNOWLEDGE
# =====================
class TransportCore:
    """Handles transportation-related queries"""
    
    def __init__(self):
        self.knowledge_base = {
            'yul_transport': {
                'response': (
                    "From Laval to YUL Airport:\n"
                    "1. Taxi/Uber: 40-60$ CAD (35-50 mins)\n"
                    "2. 747 Bus: Lionel-Groulx metro (2.75$)\n"
                    "3. Airport Shuttle: 1-800-123-4567 (65$+)\n"
                    "4. Car Rental: Available at YUL"
                ),
                'keywords': ['yul', 'airport', 'transport', 'laval', 'bus']
            }
        }

    def handle_query(self, query: str) -> Dict[str, object]:
        """Process transportation requests"""
        best_match = {
            'domain': 'transport',
            'confidence': 0.0,
            'content': None,
            'sources': []
        }
        
        for entry in self.knowledge_base.values():
            matches = sum(1 for kw in entry['keywords'] if kw in query.lower())
            confidence = matches / len(entry['keywords'])
            
            if confidence > best_match['confidence']:
                best_match.update({
                    'confidence': confidence,
                    'content': entry['response'],
                    'sources': entry['keywords']
                })
        
        return best_match

# ====================
# MAIN AI CORE CLASS
# ====================
class FridayAI:
    """Central AI processing unit with modular capabilities"""
    
    def __init__(self, memory_core: MemoryCore = None):
        # Initialize core systems
        self._configure_logging()
        self._load_environment()
        self._init_components(memory_core)
        self._init_speech()
        
        # Knowledge systems
        self.domain_handlers = {
            'transport': TransportCore()
        }
        self.router = KnowledgeRouter()

    def _configure_logging(self):
        """Set up error tracking system"""
        self.logger = logging.getLogger("FridayAI")
        self.logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler('friday_activity.log')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(module)s - %(message)s'
        ))
        self.logger.addHandler(handler)

    def _load_environment(self):
        """Load configuration from environment"""
        if not load_dotenv():
            self.logger.warning("No .env file found")
        self.api_key = os.getenv("OPENAI_API_KEY")

    def _init_components(self, memory_core):
        """Initialize cognitive subsystems"""
        self.memory = memory_core or MemoryCore()
        self.emotion = EmotionCore()
        self.auto_learner = AutoLearningCore(self.memory)
        self.self_query = SelfQueryingCore(self.memory)

    def _init_speech(self):
        """Set up text-to-speech engine"""
        try:
            self.voice = pyttsx3.init()
            self.voice.setProperty('rate', 150)
            self.voice.setProperty('voice', 'english')
        except Exception as e:
            self.logger.error(f"Voice init failed: {str(e)}")
            self.voice = None

    def _check_memory(self, query: str) -> Optional[Dict]:
        """Search memory for relevant information"""
        try:
            # Clean and search
            clean_query = query.strip().lower().replace(' ', '_')
            
            # Direct match
            if memory := self.memory.get_fact(clean_query):
                return {
                    'domain': 'memory',
                    'confidence': 1.0,
                    'content': f"I remember: {clean_query.replace('_', ' ')} = {memory}",
                    'sources': ['direct_memory']
                }
            
            # Fuzzy match
            matches = get_close_matches(clean_query, self.memory.memory.keys(), n=1, cutoff=0.6)
            if matches:
                return {
                    'domain': 'memory',
                    'confidence': 0.7,
                    'content': f"Related memory: {matches[0].replace('_', ' ')} = {self.memory.get_fact(matches[0])}",
                    'sources': ['fuzzy_memory']
                }
                
            return None
            
        except Exception as e:
            self.logger.error(f"Memory check error: {str(e)}")
            return None

    def respond_to(self, user_input: str) -> Dict[str, object]:
        """Main interface for processing user queries"""
        try:
            # First check memory
            if memory_response := self._check_memory(user_input):
                return self._enhance_response(memory_response, user_input)

            # Then check domain knowledge
            domain = self.router.detect_domain(user_input)
            if domain in self.domain_handlers:
                domain_response = self.domain_handlers[domain].handle_query(user_input)
                return self._enhance_response(domain_response, user_input)

            # Fallback to general response
            return self._enhance_response({
                'domain': 'general',
                'confidence': 0.0,
                'content': "I'm still learning about that. Can you explain more?",
                'sources': []
            }, user_input)

        except Exception as e:
            self.logger.error(f"Processing error: {str(e)}")
            return {
                'status': 'error',
                'content': "System temporarily unavailable",
                'error_code': 500
            }

    def _enhance_response(self, response: Dict, query: str) -> Dict:
        """Add contextual metadata to responses"""
        return {
            **response,
            'emotional_tone': self.emotion.analyze(query),
            'processing_time': datetime.now().isoformat(),
            'query_type': response['domain'],
            'source_confidence': response['confidence']
        }

    def speak(self, text: str):
        """Convert text to speech"""
        if self.voice:
            try:
                self.voice.say(text)
                self.voice.runAndWait()
            except Exception as e:
                self.logger.error(f"Speech failed: {str(e)}")

# ================
# COMMAND LINE UI
# ================
if __name__ == "__main__":
    ai = FridayAI()
    print("Friday AI Console - Type 'exit' to quit")
    
    while True:
        try:
            query = input("\nYou: ").strip()
            if query.lower() in ['exit', 'quit']:
                break
                
            response = ai.respond_to(query)
            print(f"\nFriday: {response['content']}")
            
        except KeyboardInterrupt:
            print("\nSession ended")
            break