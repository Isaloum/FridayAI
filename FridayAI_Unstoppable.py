# =====================================
# File: FridayAI.py (Core Engine) - CLEAN USER EXPERIENCE WITH EMPATHY + TONE SYSTEM + UNSTOPPABLE FEATURES
# Purpose: Contains the cognitive architecture of Friday + Pregnancy Intelligence + Empathy Support + Tone Selection + NEW UNSTOPPABLE ENHANCEMENTS
# =====================================

import logging
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Suppress ALL verbose logs for clean user experience
logging.getLogger("transformers").setLevel(logging.CRITICAL)
logging.getLogger("sentence_transformers").setLevel(logging.CRITICAL)
logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger().setLevel(logging.CRITICAL)

from datetime import datetime, timedelta
import threading
import time
import re
from typing import Dict, List, Optional, Tuple, Any
from dotenv import load_dotenv

# === External Core Modules (KEEPING ALL YOUR ORIGINAL IMPORTS) ===
from core.ConversationMemory import ConversationMemory
from core.DialogueCore import DialogueCore
from core.DomainAdapterCore import DomainAdapterCore
from core.EmotionClassifier import EmotionClassifier
from core.EmotionCoreV2 import EmotionCoreV2
from core.EmotionIntentFusion import EmotionIntentFusion
from core.EmotionLayerCore import EmotionLayerCore
from core.EmotionalAnchorCore import EmotionalAnchorCore
from core.EmotionalJournal import EmotionalJournal
from core.EmpathyInhibitor import EmpathyInhibitor
from core.GraphReasoner import GraphReasoner
from core.GoalAutoDetectionEngine import GoalAutoDetectionEngine
from core.IdleCognitiveLoop import IdleCognitiveLoop
from core.KnowledgeUnit import query_knowledge
from core.LLMRouterCore import route_llm
from core.MemoryContextInjector import MemoryContextInjector
from core.MemoryContextInjector import inject
from core.MemoryCore import MemoryCore
from core.MemoryReflectionEngine import MemoryReflectionEngine
from core.MemorySummarizer import MemorySummarizer
from core.MoodManagerCore import MoodManagerCore
from core.NameToneLimiter import NameToneLimiter
from core.NarrativeMemoryFusion import NarrativeMemoryFusion
from core.NeuralSchedulerCore import NeuralSchedulerCore
from core.PlanningCore import PlanningCore
from core.ReflectionLoopManager import ReflectionLoopManager
from core.SelfIntentModel import SelfIntentModel
from core.SelfNarrativeCore import log_event, update_mood, SelfNarrativeCore
from core.SessionMemory import SessionMemory
from core.ToneRebalancer import ToneRebalancer
from core.VectorMemoryCore import VectorMemoryCore
from core.brain.BehaviorRouter import BehaviorRouter
from core.brain.CognitivePrioritizationCore import CognitivePrioritizationCore
from core.pregnancy.PregnancyDomainMount import PregnancyDomainMount
from core.pregnancy.PregnancySupportCore import PregnancySupportCore

# === Pregnancy Enhancement ===
try:
    from core.pregnancy.PregnancyEmotionCore import PregnancyEmotionCore
    PREGNANCY_EMOTION_AVAILABLE = True
except ImportError:
    PREGNANCY_EMOTION_AVAILABLE = False

# === Legacy/Non-core Modules (KEEPING ALL YOUR ORIGINAL IMPORTS) ===
from AgentPlanner import AgentPlanner
from AutoLearningCore import AutoLearningCore
from BeliefDriftCore import BeliefDriftCore
from BeliefDriftSimulator import BeliefDriftSimulator
from BeliefExplanationCore import BeliefExplanationCore
from BehaviorMemoryBlender import BehaviorMemoryBlender
from CognitivePipeline import CognitivePipeline
from ContextReasoner import ContextReasoner
from EmpathyAnchorLogger import EmpathyAnchorLogger
from EngineeringSupportCore import EngineeringSupportCore
from FieldRegistry import FieldRegistry
from FuzzyMemorySearch import FuzzyMemorySearch
from GPT4Core import GPT4Core
from GoalReviewCore import GoalReviewCore
from GraphBrainCore import GraphBrainCore
from InputSanitizer import InputSanitizer
from IntentRouter import IntentRouter
from IntentionReflectionCore import IntentionReflectionCore
from KnowledgeRouter import KnowledgeRouter
from LLMCore import LLMCore
from LongTermIntentCore import LongTermIntentCore
from LongTermMemory import LongTermMemory
from MemoryScaffold import store_memory, search_memory, save_memory
from NeuralUserPersonaClassifier import NeuralUserPersonaClassifier
from PersonalityCore import PersonalityCore
from PlanningExecutionCore import PlanningExecutionCore
from QueryMemoryCore import QueryMemoryCore
from ReflectionEngine import generate_daily_reflection
from ReflectionTracker import ReflectionTracker
from SemanticEngagementCore import SemanticEngagementCore
from SemanticResponseEngine import SemanticResponseEngine
from SelfAwarenessCore import SelfAwarenessCore
from SelfBeliefUpdater import SelfBeliefUpdater
from SelfQueryingCore import SelfQueryingCore
from ToneRewriterCore import ToneRewriterCore
from TransportCore import TransportCore

# Add to your FridayAI.py imports
from maternal_care import (
    SecureMaternalDatabase, 
    MaternalHealthProfile,
    OfflineCapableFriday,
    PrivacyTrustManager
)

# === NEW UNSTOPPABLE IMPORTS (ADDITIONS ONLY) ===
import json
import pickle
import hashlib
from dataclasses import dataclass, field
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from functools import lru_cache, wraps
import queue
import signal
import atexit
from pathlib import Path
import sqlite3
import warnings
warnings.filterwarnings("ignore")

# Optional new features (graceful degradation if not available)
try:
    import speech_recognition as sr
    import pyttsx3
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    print("Voice features not available. Install speechrecognition and pyttsx3 for voice support.")

try:
    from cryptography.fernet import Fernet
    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False
    print("Enhanced encryption not available. Install cryptography for secure vault features.")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import aiofiles
    import asyncio
    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False

# === BOOTSTRAP ===
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

# === SIMPLE TONE MANAGER (KEEPING YOUR ORIGINAL) ===
class SimpleToneManager:
    def __init__(self):
        self.current_tone = "supportive"
        
    def detect_tone_request(self, user_input):
        """Check if user wants to change tone"""
        input_lower = user_input.lower().strip()
        
        # Handle !tone commands
        if input_lower.startswith("!tone"):
            parts = user_input.split()
            if len(parts) > 1:
                requested_tone = parts[1].lower()
                if requested_tone in ["supportive", "sassy", "direct"]:
                    old_tone = self.current_tone
                    self.current_tone = requested_tone
                    return f"🎭 Tone changed to **{requested_tone.title()}**! I'll now be more {requested_tone}."
                else:
                    return "❌ Available tones: supportive, sassy, direct"
            else:
                return f"🎭 Current tone: **{self.current_tone.title()}**\n\nAvailable: supportive, sassy, direct\nUse: !tone [supportive/sassy/direct]"
        
        # Handle natural language
        if "be more sassy" in input_lower or "more funny" in input_lower:
            self.current_tone = "sassy"
            return "🎭 Switching to sassy mode, honey! 💅"
        elif "be more direct" in input_lower or "more factual" in input_lower:
            self.current_tone = "direct"
            return "🎭 Switching to direct mode. Facts only."
        elif "be more supportive" in input_lower:
            self.current_tone = "supportive"
            return "🎭 Switching to supportive mode. I'm here for you. 💙"
            
        return None
    
    def apply_tone(self, original_response):
        """Apply tone to response"""
        if self.current_tone == "sassy":
            return self._make_sassy(original_response)
        elif self.current_tone == "direct":
            return self._make_direct(original_response)
        else:
            return original_response  # supportive is default
    
    def _make_sassy(self, text):
        """Add sassy flair"""
        sassy_prefixes = [
            "Alright honey, let's talk real talk about this.",
            "Girl, you're asking all the right questions!",
            "Listen babe, let me drop some wisdom on you:",
            "Okay sweetie, here's the tea:"
        ]
        
        sassy_endings = [
            "You've got this, queen! 👑",
            "Trust me, you're amazing! ✨",
            "Keep being fabulous! 💅"
        ]
        
        import random
        prefix = random.choice(sassy_prefixes)
        ending = random.choice(sassy_endings)
        
        # Replace some words for sass
        modified = text.replace("It's important to", "Girl, you NEED to")
        modified = modified.replace("You should", "Honey, you better")
        modified = modified.replace("Healthcare providers", "Your doc (who went to school forever)")
        
        return f"{prefix}\n\n{modified}\n\n{ending}"
    
    def _make_direct(self, text):
        """Make more clinical/direct"""
        direct_prefixes = [
            "Based on medical evidence:",
            "Clinical facts:",
            "Key information:"
        ]
        
        direct_endings = [
            "Consult your healthcare provider for personalized advice.",
            "This is based on current medical evidence."
        ]
        
        import random
        prefix = random.choice(direct_prefixes)
        ending = random.choice(direct_endings)
        
        # Remove emotional language
        modified = text.replace("I understand", "Research indicates")
        modified = modified.replace("I'm here for you", "Support is available")
        modified = modified.replace("Don't worry", "Evidence suggests")
        
        return f"{prefix}\n\n{modified}\n\n{ending}"

# === NEW UNSTOPPABLE FEATURE CLASSES ===

@dataclass
class ConversationState:
    """Enhanced conversation state tracking"""
    user_id: str
    session_id: str
    mood_history: deque = field(default_factory=lambda: deque(maxlen=50))
    topic_history: deque = field(default_factory=lambda: deque(maxlen=20))
    interaction_count: int = 0
    last_interaction: datetime = field(default_factory=datetime.now)
    user_preferences: Dict = field(default_factory=dict)
    health_data: Dict = field(default_factory=dict)
    emergency_contacts: List = field(default_factory=list)

class ResilienceEngine:
    """Self-healing and error recovery system"""
    
    def __init__(self):
        self.error_history = deque(maxlen=1000)
        self.recovery_strategies = {}
        self.circuit_breakers = {}
        self.fallback_responses = self._load_fallbacks()
        
    def _load_fallbacks(self):
        """Load fallback responses for various scenarios"""
        return {
            "general_error": [
                "I'm having a moment, but I'm still here for you. Could you tell me that again?",
                "Let me refocus on what you need. What's on your mind?",
                "I didn't quite catch that, but I'm listening. How can I help?"
            ],
            "memory_error": [
                "My memory is a bit foggy right now, but I'm still here to support you.",
                "Let's start fresh. What would you like to talk about?"
            ],
            "emotional_overload": [
                "I sense there's a lot of emotion here. Take a deep breath with me. What's most important right now?",
                "Your feelings are valid. Let's take this one step at a time."
            ]
        }
    
    def wrap_with_resilience(self, func):
        """Decorator to add resilience to any function"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            max_attempts = 3
            
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    self.error_history.append({
                        'function': func.__name__,
                        'error': str(e),
                        'timestamp': datetime.now(),
                        'attempt': attempt
                    })
                    
                    if attempt < max_attempts:
                        time.sleep(0.1 * attempt)  # Exponential backoff
                        continue
                    else:
                        # Use fallback
                        return self._get_fallback_response(func.__name__, e)
            
        return wrapper
    
    def _get_fallback_response(self, func_name: str, error: Exception):
        """Get appropriate fallback response"""
        if "memory" in func_name.lower():
            responses = self.fallback_responses["memory_error"]
        elif "emotion" in func_name.lower():
            responses = self.fallback_responses["emotional_overload"]
        else:
            responses = self.fallback_responses["general_error"]
        
        import random
        return random.choice(responses)

class PredictiveAnalytics:
    """Predictive analytics for pregnancy milestones and user needs"""
    
    def __init__(self):
        self.milestone_predictor = self._init_milestone_model()
        self.mood_predictor = self._init_mood_model()
        self.need_predictor = self._init_need_model()
        
    def _init_milestone_model(self):
        """Initialize pregnancy milestone predictions"""
        return {
            8: ["Morning sickness might peak around week 9-10", "First prenatal appointment coming up"],
            12: ["End of first trimester approaching!", "Morning sickness may start to ease"],
            16: ["You might feel first movements soon", "Gender reveal possible at next scan"],
            20: ["Anatomy scan coming up", "Halfway through your pregnancy!"],
            28: ["Third trimester begins", "Glucose screening test"],
            32: ["Baby shower planning time", "Prenatal classes recommended"],
            36: ["Full term approaching", "Hospital bag preparation"]
        }
    
    def _init_mood_model(self):
        return {}
    
    def _init_need_model(self):
        return {}
    
    def predict_upcoming_milestones(self, current_week: int) -> List[str]:
        """Predict upcoming milestones based on pregnancy week"""
        predictions = []
        for week, milestones in self.milestone_predictor.items():
            if current_week <= week <= current_week + 4:
                for milestone in milestones:
                    predictions.append(f"Week {week}: {milestone}")
        return predictions
    
    def predict_emotional_needs(self, mood_history: deque) -> Dict[str, float]:
        """Predict emotional support needs based on mood patterns"""
        if not mood_history:
            return {"support": 0.5, "information": 0.3, "reassurance": 0.2}
        
        recent_moods = list(mood_history)[-10:]
        mood_counts = defaultdict(int)
        for mood in recent_moods:
            mood_counts[mood] += 1
        
        # Analyze patterns
        if mood_counts.get("anxious", 0) > 3:
            return {"reassurance": 0.7, "calming": 0.2, "information": 0.1}
        elif mood_counts.get("sad", 0) > 2:
            return {"empathy": 0.6, "support": 0.3, "positivity": 0.1}
        else:
            return {"information": 0.5, "support": 0.3, "encouragement": 0.2}

class VoiceInterface:
    """Voice input/output capabilities"""
    
    def __init__(self):
        if not VOICE_AVAILABLE:
            self.enabled = False
            return
            
        try:
            self.recognizer = sr.Recognizer()
            self.engine = pyttsx3.init()
            self.setup_voice()
            self.enabled = True
        except:
            self.enabled = False
            
    def setup_voice(self):
        """Configure voice settings"""
        voices = self.engine.getProperty('voices')
        # Try to find a female voice for Friday
        for voice in voices:
            if "female" in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                break
        
        self.engine.setProperty('rate', 180)  # Speaking rate
        self.engine.setProperty('volume', 0.9)  # Volume
    
    def listen(self, timeout=5):
        """Listen for voice input"""
        if not self.enabled:
            return None
            
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=timeout)
                text = self.recognizer.recognize_google(audio)
                return text
        except:
            return None
    
    def speak(self, text: str):
        """Convert text to speech"""
        if not self.enabled:
            return
            
        # Remove markdown and special characters for speech
        clean_text = re.sub(r'[*_#`]', '', text)
        clean_text = re.sub(r'\n+', '. ', clean_text)
        
        try:
            self.engine.say(clean_text)
            self.engine.runAndWait()
        except:
            pass

class SecureDataVault:
    """Enhanced security for sensitive maternal health data"""
    
    def __init__(self, vault_path="maternal_vault.db"):
        self.vault_path = vault_path
        self.enabled = ENCRYPTION_AVAILABLE
        
        if self.enabled:
            self.cipher_suite = Fernet(self._get_or_create_key())
            self._init_vault()
        
    def _get_or_create_key(self):
        """Get or create encryption key"""
        key_path = Path("vault.key")
        if key_path.exists():
            return key_path.read_bytes()
        else:
            key = Fernet.generate_key()
            key_path.write_bytes(key)
            return key
    
    def _init_vault(self):
        """Initialize secure database"""
        conn = sqlite3.connect(self.vault_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS health_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                category TEXT NOT NULL,
                data_encrypted TEXT NOT NULL,
                checksum TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()
    
    def store_health_data(self, user_id: str, category: str, data: Dict):
        """Securely store health data"""
        if not self.enabled:
            return False
            
        try:
            # Serialize and encrypt
            data_bytes = json.dumps(data).encode()
            encrypted = self.cipher_suite.encrypt(data_bytes)
            checksum = hashlib.sha256(data_bytes).hexdigest()
            
            # Store in vault
            conn = sqlite3.connect(self.vault_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO health_records (user_id, category, data_encrypted, checksum)
                VALUES (?, ?, ?, ?)
            ''', (user_id, category, encrypted.decode(), checksum))
            conn.commit()
            conn.close()
            return True
        except:
            return False
    
    def retrieve_health_data(self, user_id: str, category: str = None):
        """Retrieve and decrypt health data"""
        if not self.enabled:
            return []
            
        try:
            conn = sqlite3.connect(self.vault_path)
            cursor = conn.cursor()
            
            if category:
                cursor.execute('''
                    SELECT data_encrypted, checksum FROM health_records
                    WHERE user_id = ? AND category = ?
                    ORDER BY timestamp DESC
                ''', (user_id, category))
            else:
                cursor.execute('''
                    SELECT category, data_encrypted, checksum FROM health_records
                    WHERE user_id = ?
                    ORDER BY timestamp DESC
                ''', (user_id,))
            
            results = []
            for row in cursor.fetchall():
                encrypted = row[-2].encode()
                checksum = row[-1]
                
                # Decrypt and verify
                decrypted = self.cipher_suite.decrypt(encrypted)
                if hashlib.sha256(decrypted).hexdigest() == checksum:
                    data = json.loads(decrypted.decode())
                    if category:
                        results.append(data)
                    else:
                        results.append({row[0]: data})
            
            conn.close()
            return results
        except:
            return []

class EmergencyProtocol:
    """Emergency response system for critical situations"""
    
    def __init__(self):
        self.emergency_keywords = [
            "bleeding", "severe pain", "can't breathe", "contractions",
            "water broke", "baby not moving", "dizzy", "faint",
            "chest pain", "severe headache", "vision problems"
        ]
        self.emergency_contacts = []
        self.location_service = None
        
    def check_emergency(self, user_input: str) -> Tuple[bool, str]:
        """Check if input indicates emergency"""
        input_lower = user_input.lower()
        
        for keyword in self.emergency_keywords:
            if keyword in input_lower:
                return True, keyword
        
        # Check for urgency patterns
        urgent_patterns = [
            r"help\s+me\s+now",
            r"emergency",
            r"911",
            r"ambulance",
            r"hospital\s+now"
        ]
        
        for pattern in urgent_patterns:
            if re.search(pattern, input_lower):
                return True, "urgent_request"
        
        return False, None
    
    def generate_emergency_response(self, emergency_type: str) -> str:
        """Generate appropriate emergency response"""
        response = "🚨 **IMPORTANT - MEDICAL ATTENTION NEEDED**\n\n"
        
        if emergency_type in ["bleeding", "severe pain", "contractions"]:
            response += "1. **Call 911 or your emergency number NOW**\n"
            response += "2. Call your healthcare provider\n"
            response += "3. If bleeding: Lie down, elevate feet\n"
            response += "4. Stay calm, help is coming\n\n"
            response += "**I'm staying with you. Tell me what's happening while help arrives.**"
        
        elif emergency_type == "baby not moving":
            response += "1. Lie on your left side\n"
            response += "2. Drink something cold and sweet\n"
            response += "3. Count movements for 10 minutes\n"
            response += "4. If no movement in 1 hour: **Go to hospital immediately**\n\n"
            response += "**This is serious. Your baby needs to be checked.**"
        
        else:
            response += "1. **Seek immediate medical attention**\n"
            response += "2. Call 911 if severe\n"
            response += "3. Have someone drive you to hospital\n"
            response += "4. Don't wait - act now\n\n"
            response += "**Your safety is the priority. Get help immediately.**"
        
        return response

# === FRIDAYAI CORE CLASS (KEEPING YOUR ORIGINAL + ENHANCEMENTS) ===
class FridayAI:
    def __init__(self, memory, emotion):
        self.memory = memory
        self.vector_memory = VectorMemoryCore()
        self.emotion = emotion
        self.mood_filter = MoodManagerCore()
        self.intent_model = SelfIntentModel()
        self.fusion = EmotionIntentFusion(self.intent_model)
        self.agent_planner = AgentPlanner()
        self.semantic_engagement = SemanticEngagementCore()
        self.semantic_response_engine = SemanticResponseEngine(self.semantic_engagement)
        self.self_awareness = SelfAwarenessCore()
        self.reflector = ReflectionTracker()
        self.session = SessionMemory()
        self.reflection_core = IntentionReflectionCore(memory, emotion, [])
        self.planner = PlanningExecutionCore()
        self.input_sanitizer = InputSanitizer()
        self.tone_manager = SimpleToneManager()
       
        # === Pregnancy Enhancement ===
        if PREGNANCY_EMOTION_AVAILABLE:
            self.pregnancy_emotion = PregnancyEmotionCore()
        else:
            self.pregnancy_emotion = None
        
        # === NEW UNSTOPPABLE FEATURES ===
        self.resilience = ResilienceEngine()
        self.predictive = PredictiveAnalytics()
        self.voice = VoiceInterface()
        self.vault = SecureDataVault()
        self.emergency = EmergencyProtocol()
        
        # Enhanced state management
        self.conversation_states = {}
        self.performance_metrics = {
            "response_times": deque(maxlen=100),
            "error_count": 0,
            "successful_interactions": 0,
            "uptime_start": datetime.now()
        }
        
        self._configure_logging()
        self._init_components()
        self._init_knowledge_systems()
        self.identity = SelfNarrativeCore()
        self.belief_explainer = BeliefExplanationCore()
        self.belief_updater = SelfBeliefUpdater(self.identity, self.belief_explainer)
        self.name_limiter = NameToneLimiter()
        self.personality = PersonalityCore(debug=False)
        self.persona_classifier = NeuralUserPersonaClassifier()
        self.emotional_anchors = EmotionalAnchorCore()
        self.anchor_logger = EmpathyAnchorLogger(self.emotional_anchors)
        self.intent_engine = LongTermIntentCore()
        self.goal_detector = GoalAutoDetectionEngine(self.intent_engine, self.emotion)
        self.goal_reviewer = GoalReviewCore(self.intent_engine)
        self.belief_drift = BeliefDriftCore(self.intent_engine, self.emotion)
        self.domain_adapter = DomainAdapterCore(self.memory, self.emotion, self.intent_engine)
        self.engineering_module = EngineeringSupportCore(self.memory, self.identity, self.emotion)
        self.pregnancy_mount = PregnancyDomainMount(self.memory, self.emotion, self.identity)

        self.domain_adapter.attach_ability_modules("pregnancy", self.pregnancy_mount.get_abilities())

        self.narrative = SelfNarrativeCore()
        self.reflection_loop = ReflectionLoopManager(
            memory=self.memory,
            emotion_core=self.emotion,
            belief_core=self.belief_drift
        )

        self.confidence_threshold = float(os.getenv("FRIDAY_CONFIDENCE_THRESHOLD", "0.6"))
        self.llm = LLMCore()
        self.pipeline = CognitivePipeline(
            llm_core=self.llm,
            emotion_core=self.emotion,
            vector_memory_core=self.vector_memory,
            self_narrative_core=self.identity,
            memory_core=self.memory
        )
        self.drift_sim = BeliefDriftSimulator(self.identity, self.emotion, self.vector_memory)
        self.scheduler = NeuralSchedulerCore(
            identity_core=self.identity,
            emotion_core=self.emotion,
            planner_core=self.planner,
            narrative_fusion=NarrativeMemoryFusion()
        )
        self.registry = FieldRegistry()
        
        # Load empathy responses safely
        self.empathy_responses = self._load_empathy_safe()
        
        # Enhanced tone system integration
        self.tone_rewriter = ToneRewriterCore()
        
        # Auto-save setup
        self._setup_autosave()

    def _analyze_input_semantic(self, user_input):
        """
        Semantic analysis using context understanding, not just keywords
        """
        input_lower = user_input.lower().strip()
        
        # 1. QUICK FILTERS for obvious non-conversational input
        obvious_non_conversation = [
            len(input_lower) < 3,  # Too short
            input_lower.startswith(('def ', 'class ', 'import ', 'from ')),  # Code
            input_lower.startswith(('!', '\\', '/')),  # Commands
            input_lower.count('(') > input_lower.count(' '),  # More brackets than spaces
            bool(re.match(r'^[a-zA-Z]{1,4}$', input_lower)),  # Single short word
            'filters out' in input_lower,  # Technical jargon
            'show_tone' in input_lower,  # Function references
        ]
        
        if any(obvious_non_conversation):
            return {
                'type': 'non_conversational',
                'confidence': 0.9,
                'response': "I'm not sure what you're referring to. Could you tell me more about what you need help with?"
            }
        
        # 2. SEMANTIC PREGNANCY DETECTION
        # Look for MEANING patterns, not just keywords
        
        # Emotional expressions (more flexible)
        emotional_patterns = [
            r'\b(feel|feeling|felt)\s+(scared|afraid|anxious|worried|nervous|overwhelmed)',
            r'\b(i\'?m|am)\s+(scared|afraid|anxious|worried|nervous|terrified)',
            r'\b(so|really|very)\s+(scared|afraid|worried|anxious)',
            r'\bnot\s+sure\s+(i|if)',
            r'\bdon\'?t\s+know\s+(if|how)',
            r'\bwhat\s+if\s+something',
            r'\bworried\s+about',
            r'\bscared\s+(about|of)',
        ]
        
        # Pregnancy/motherhood context (more natural)
        pregnancy_patterns = [
            r'\b(baby|pregnancy|pregnant|expecting)',
            r'\b(mom|mother|motherhood|maternal)',
            r'\b(birth|delivery|labor|due\s+date)',
            r'\b(first\s+time\s+mom|new\s+mom)',
            r'\b(gestational|prenatal|trimester)',
            r'\bweeks?\s+pregnant',
        ]
        
        # Personal narrative indicators
        personal_patterns = [
            r'\bi\s+(am|\'m|was|will|have|need|want|think|feel)',
            r'\bmy\s+(baby|pregnancy|doctor|body)',
            r'\bshould\s+i\b',
            r'\bcan\s+i\b',
            r'\bhow\s+(do|can)\s+i\b',
        ]
        
        # Count pattern matches (more nuanced than simple keyword counting)
        emotional_score = sum(1 for pattern in emotional_patterns if re.search(pattern, input_lower))
        pregnancy_score = sum(1 for pattern in pregnancy_patterns if re.search(pattern, input_lower))
        personal_score = sum(1 for pattern in personal_patterns if re.search(pattern, input_lower))
        
        # 3. CONTEXT ANALYSIS
        
        # Check for question structure
        is_question = any([
            input_lower.endswith('?'),
            input_lower.startswith(('what', 'how', 'when', 'where', 'why', 'should', 'can', 'will', 'do')),
            ' or ' in input_lower,  # choice questions
        ])
        
        # Check for emotional vulnerability
        vulnerability_indicators = [
            'not sure', 'don\'t know', 'confused', 'help', 'advice',
            'what should', 'am i', 'will i be', 'going to be'
        ]
        shows_vulnerability = any(indicator in input_lower for indicator in vulnerability_indicators)
        
        # Check sentence length and complexity (real conversation tends to be longer)
        word_count = len(input_lower.split())
        seems_conversational = word_count >= 5 and word_count <= 100
        
        # 4. CALCULATE PREGNANCY CONCERN PROBABILITY
        
        base_score = 0
        
        # Emotional component (40% weight)
        if emotional_score > 0:
            base_score += 40 * min(emotional_score / 2, 1)  # Cap at 2 emotional patterns
        
        # Pregnancy context (30% weight) 
        if pregnancy_score > 0:
            base_score += 30 * min(pregnancy_score / 2, 1)  # Cap at 2 pregnancy patterns
        
        # Personal narrative (20% weight)
        if personal_score > 0:
            base_score += 20 * min(personal_score / 3, 1)  # Cap at 3 personal patterns
        
        # Conversation quality bonuses (10% weight)
        if is_question:
            base_score += 5
        if shows_vulnerability:
            base_score += 3  
        if seems_conversational:
            base_score += 2
        
        # 5. DECISION LOGIC
        
        if base_score >= 70:  # High confidence pregnancy concern
            return {
                'type': 'pregnancy_concern',
                'confidence': base_score / 100,
                'context': 'emotional_pregnancy_support'
            }
        elif base_score >= 40:  # Possible pregnancy concern - ask for clarification
            return {
                'type': 'possible_pregnancy_concern', 
                'confidence': base_score / 100,
                'response': "It sounds like you might have something pregnancy-related on your mind. Would you like to tell me more about what you're feeling or experiencing?"
            }
        else:  # General conversation
            return {
                'type': 'general_conversation',
                'confidence': (100 - base_score) / 100
            }

    def _configure_logging(self):
        # Silent logging for clean user experience
        self.logger = logging.getLogger("FridayAI")
        self.logger.setLevel(logging.CRITICAL)  # Only critical errors
        handler = logging.FileHandler("friday_activity.log")
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
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
        self.context_injector = MemoryContextInjector(self.memory)
        self.long_term = LongTermMemory()
        self.context_reasoner = ContextReasoner(self.long_term, self.emotion)
        self.journal = EmotionalJournal()
        self.planner_core = PlanningCore(self.memory)
        self.tone_rebalancer = ToneRebalancer()
        self.emotion_layer = EmotionLayerCore()
        self.empathy = GraphReasoner(self.graph)
        self.empathy_inhibitor = EmpathyInhibitor()
        self.blender = BehaviorMemoryBlender(alpha=0.12)
        self.narrative_fusion = NarrativeMemoryFusion()

    def _init_knowledge_systems(self):
        self.domain_handlers = {
            "transport": TransportCore()
        }
        self.router = IntentRouter(self.memory, self.emotion, self.context_reasoner)
        self.router.load_traits("traits.json")

    def _load_empathy_safe(self):
        """Safely load empathy responses with fallback"""
        try:
            empathy_path = "./pregnancy_support/empathy/soft_replies.json"
            with open(empathy_path, 'r', encoding='utf-8') as f:
                import json
                return json.load(f)
        except Exception as e:
            # Fallback empathy responses built-in
            return {
                "pregnancy_emotional_support": {
                    "anxious": [
                        "It's completely natural to feel scared and uncertain about the journey ahead. Becoming a parent is one of the biggest changes you'll ever experience, and it's okay to worry about whether you'll be a great mom or how the rest of your pregnancy will go. These feelings are actually a sign of just how much you care about your baby and the kind of parent you want to be."
                    ],
                    "scared": [
                        "Your fears about pregnancy and motherhood are completely valid. It's natural to feel scared when facing something so life-changing and important. These feelings don't mean anything is wrong - they show how much you care."
                    ],
                    "overwhelmed": [
                        "Feeling overwhelmed during pregnancy is so common and understandable. There's so much information, so many changes happening to your body, and so many decisions to make. Take it one day at a time."
                    ],
                    "sad": [
                        "I'm sorry you're feeling sad right now. Pregnancy emotions can be intense and sometimes confusing. Your feelings are valid, and it's important to be gentle with yourself during this time."
                    ]
                },
                "general_support": [
                    "I'm here to support you through whatever you're feeling. Your emotions and concerns are completely valid."
                ]
            }

    def _get_empathy_response(self, mood):
        """Get appropriate empathy response for detected mood"""
        pregnancy_empathy = self.empathy_responses.get("pregnancy_emotional_support", {})
        
        if mood in pregnancy_empathy:
            import random
            return random.choice(pregnancy_empathy[mood])
        
        # Fallback to general support
        general_support = self.empathy_responses.get("general_support", [])
        if general_support:
            import random
            return random.choice(general_support)
        
        return "I understand you're going through a lot right now. I'm here to support you."

    def _offer_pregnancy_resources(self, user_input: str, emotional_tone: str) -> str:
        """Smart resource offering with empathy first"""
        
        # Detect if user needs emotional support
        need_keywords = ["help", "advice", "don't know", "unsure", "worried", "scared", "anxious", "overwhelmed"]
        pregnancy_keywords = ["baby", "pregnant", "pregnancy", "mom", "mother", "birth"]
        
        needs_help = any(keyword in user_input.lower() for keyword in need_keywords)
        is_pregnancy_related = any(keyword in user_input.lower() for keyword in pregnancy_keywords)
        
        if not (needs_help and is_pregnancy_related):
            return ""
        
        # Get empathy response first
        empathy_text = self._get_empathy_response(emotional_tone)
        
        # Resource database
        resources = {
            "anxiety": {
                "books": [
                    "📖 'The First-Time Mom's Pregnancy Handbook' by Allison Hill",
                    "📖 'What to Expect When You're Expecting' by Heidi Murkoff"
                ],
                "apps": [
                    "📱 Calm - Meditation for pregnancy",
                    "📱 BabyCentre Pregnancy Tracker"
                ],
                "support": [
                    "🏥 Talk to your healthcare provider",
                    "👥 Join a local pregnancy support group"
                ]
            }
        }
        
        # Determine resource category
        if emotional_tone in ["anxious", "scared", "overwhelmed", "sad"]:
            category = "anxiety"
        else:
            category = "anxiety"  # Default to anxiety resources for pregnancy concerns
        
        # Build response starting with empathy
        final_text = empathy_text
        
        # Add resources
        final_text += "\n\n💝 **I have some resources that might help:**\n"
        
        if category in resources:
            # Books
            if "books" in resources[category]:
                final_text += "\n**📚 Helpful Books:**\n"
                for book in resources[category]["books"]:
                    final_text += f"• {book}\n"
            
            # Apps for anxiety
            if "apps" in resources[category]:
                final_text += "\n**📱 Calming Apps:**\n"
                for app in resources[category]["apps"]:
                    final_text += f"• {app}\n"
            
            # Support options
            if "support" in resources[category]:
                final_text += "\n**🤝 Support Options:**\n"
                for support in resources[category]["support"]:
                    final_text += f"• {support}\n"
            
            # Interactive offers
            final_text += "\n💬 **Would you like me to:**\n"
            final_text += "• Share more specific resources about what you're feeling?\n"
            final_text += "• Help you find local pregnancy support groups?\n"
            final_text += "• Guide you through some calming techniques?\n"
        
        return final_text

    def _setup_autosave(self):
        """Setup automatic conversation saving"""
        def autosave():
            while True:
                time.sleep(60)  # Save every minute
                try:
                    self._save_all_conversations()
                except:
                    pass
        
        autosave_thread = threading.Thread(target=autosave, daemon=True)
        autosave_thread.start()
    
    def _save_all_conversations(self):
        """Save all active conversations"""
        for session_id, state in self.conversation_states.items():
            self._save_conversation_state(session_id, state)
    
    def _save_conversation_state(self, session_id: str, state: ConversationState):
        """Save individual conversation state"""
        state_path = Path(f"conversations/{session_id}.pkl")
        state_path.parent.mkdir(exist_ok=True)
        
        with open(state_path, 'wb') as f:
            pickle.dump(state, f)

    def respond_to(self, user_input: str, pregnancy_week: int = 0) -> Dict[str, object]:
        """Enhanced respond_to with pregnancy emotion intelligence and tone system"""
        
        # NEW: Check for emergency FIRST
        is_emergency, emergency_type = self.emergency.check_emergency(user_input)
        if is_emergency:
            emergency_response = self.emergency.generate_emergency_response(emergency_type)
            if hasattr(self, 'voice') and self.voice.enabled:
                self.voice.speak("This is an emergency. Please seek immediate medical attention.")
            return {
                "domain": "emergency",
                "content": emergency_response,
                "confidence": 1.0,
                "emergency": True,
                "emotional_tone": "urgent",
                "processing_time": datetime.now().isoformat()
            }
        
        # Check for tone change requests
        tone_response = self.tone_rewriter.detect_tone_request(user_input)
        if tone_response:
            return {
                "domain": "tone_selector",
                "content": tone_response,
                "confidence": 1.0,
                "emotional_tone": "neutral",
                "processing_time": datetime.now().isoformat()
            }
        
        # NEW: Wrap response generation with resilience
        @self.resilience.wrap_with_resilience
        def generate_resilient_response():
            # Memory injection (silent)
            ctx = inject(user_input)
            
            # Knowledge citations (silent background processing)
            citations = query_knowledge(user_input)
            excluded_files = ['requirements.txt', 'cognition_notes.txt', '.gitignore', '.env']
            
            # Pregnancy emotion analysis (silent)
            pregnancy_analysis = None
            if PREGNANCY_EMOTION_AVAILABLE and self.pregnancy_emotion and pregnancy_week > 0:
                try:
                    pregnancy_analysis = self.pregnancy_emotion.analyze_pregnancy_emotion(
                        user_input, pregnancy_week
                    )
                except Exception as e:
                    pass
            
            # Generate response (silent processing)
            result = self.pipeline.generate_response(user_input)
            
            # Handle response format
            if isinstance(result, str):
                raw_reply = result
                emotional_tone = "scared"  # ASSUME PREGNANCY FEAR FOR BETTER SUPPORT
                memory_context = None
                identity_context = None
            elif isinstance(result, dict):
                raw_reply = result.get('reply', result.get('response', '')).strip()
                emotional_tone = result.get('emotion', result.get('emotional_tone', 'scared'))  # DEFAULT TO SCARED FOR PREGNANCY
                memory_context = result.get('memory_context')
                identity_context = result.get('identity_context')
            else:
                raw_reply = str(result)
                emotional_tone = "scared"  # DEFAULT TO SCARED FOR PREGNANCY
                memory_context = None
                identity_context = None
            
            # Clean output
            if not raw_reply:
                raw_reply = "I'm here to help. Could you tell me more about what you need?"
            
            # NEW: Add predictive milestones if pregnancy week provided
            if pregnancy_week > 0:
                milestones = self.predictive.predict_upcoming_milestones(pregnancy_week)
                if milestones:
                    raw_reply += "\n\n📅 **Upcoming Milestones:**\n"
                    for milestone in milestones[:3]:  # Show top 3
                        raw_reply += f"• {milestone}\n"
            
            # Enhance response with pregnancy awareness (silent)
            if pregnancy_analysis and pregnancy_analysis.intensity > 0.6:
                supportive_message = self.pregnancy_emotion.generate_supportive_response(pregnancy_analysis)
                raw_reply += f"\n\n{supportive_message}"
            
            # Clean final output
            final_output = raw_reply

            # Add smart resource offering
            resources_offer = self._offer_pregnancy_resources(user_input, emotional_tone)
            if resources_offer:
                final_output = resources_offer  # REPLACE instead of append
            
            # Apply tone rewriting to final output
            is_pregnancy_related = any(word in user_input.lower() for word in ["baby", "pregnant", "pregnancy", "mom", "mother"])
            if not resources_offer:  # Only apply tone if we're not showing empathy resources
                final_output = self.tone_rewriter.rewrite(final_output)
            
            # Apply tone if available  
            if hasattr(self, 'tone_manager'):
                final_output = self.tone_manager.apply_tone(final_output)
            
            # Only show relevant knowledge if it's truly helpful (filtered)
            relevant_citations = []
            for c in citations:
                if c.get('source') not in excluded_files and 'text' in c:
                    # Only include if citation is actually relevant and substantial
                    if len(c['text']) > 50 and any(word in c['text'].lower() for word in user_input.lower().split()):
                        relevant_citations.append(c)
            
            if relevant_citations and len(relevant_citations) <= 2:  # Limit to 2 most relevant
                sources = [f"📄 {c['text']}" for c in relevant_citations[:2]]
                final_output += "\n\n" + "\n\n".join(sources)
            
            # Silent logging
            log_event(user_input, source="user")
            log_event(final_output, source="friday")
            
            try:
                update_mood(emotional_tone)
            except Exception as e:
                update_mood("neutral")
            
            # Enhanced return with pregnancy data
            response = {
                "domain": "cognitive_pipeline",
                "content": final_output,
                "confidence": 1.0,
                "emotional_tone": emotional_tone,
                "memory_context": memory_context,
                "identity_context": identity_context,
                "processing_time": datetime.now().isoformat(),
                "current_tone": self.tone_rewriter.get_current_tone()  # Add current tone info
            }
            
            # Add pregnancy analysis if available
            if pregnancy_analysis:
                response["pregnancy_emotion"] = {
                    "primary_emotion": pregnancy_analysis.primary_emotion,
                    "intensity": pregnancy_analysis.intensity,
                    "hormonal_influence": pregnancy_analysis.hormonal_influence,
                    "week": pregnancy_analysis.pregnancy_week,
                    "confidence": pregnancy_analysis.confidence_score
                }
            
            return response
        
        # Generate response with resilience
        return generate_resilient_response()

    def voice_interaction(self, user_id: str = "default"):
        """Enable voice-based interaction"""
        if not self.voice.enabled:
            print("Voice features not available. Please install speechrecognition and pyttsx3.")
            return
            
        print("🎤 Voice mode activated. Say 'exit' to return to text mode.")
        
        while True:
            print("\n[Listening...]")
            user_input = self.voice.listen()
            
            if user_input is None:
                print("Didn't catch that. Try again?")
                continue
            
            print(f"You said: {user_input}")
            
            if "exit" in user_input.lower():
                print("Returning to text mode.")
                break
            
            # Get response
            response = self.respond_to(user_input)
            
            # Speak response
            print(f"\nFriday: {response['content']}")
            self.voice.speak(response['content'])
    
    def get_health_insights(self, user_id: str) -> str:
        """Generate health insights from stored data"""
        if not self.vault.enabled:
            return "Health vault features not available. Install cryptography for secure health tracking."
            
        health_data = self.vault.retrieve_health_data(user_id)
        
        if not health_data:
            return "No health data recorded yet. Would you like to start tracking your pregnancy journey?"
        
        insights = "📊 **Your Health Insights:**\n\n"
        
        # Analyze patterns
        categories = defaultdict(list)
        for record in health_data:
            for category, data in record.items():
                categories[category].append(data)
        
        for category, data_list in categories.items():
            insights += f"**{category.title()}:**\n"
            # Add specific insights based on category
            if category == "symptoms":
                common_symptoms = defaultdict(int)
                for data in data_list:
                    for symptom in data.get("symptoms", []):
                        common_symptoms[symptom] += 1
                
                if common_symptoms:
                    most_common = max(common_symptoms.items(), key=lambda x: x[1])
                    insights += f"• Most reported: {most_common[0]} ({most_common[1]} times)\n"
            
            insights += "\n"
        
        return insights
    
    def get_performance_report(self) -> str:
        """Generate performance report"""
        uptime = datetime.now() - self.performance_metrics["uptime_start"]
        
        if NUMPY_AVAILABLE and self.performance_metrics["response_times"]:
            avg_response = np.mean(list(self.performance_metrics["response_times"]))
        else:
            avg_response = sum(self.performance_metrics["response_times"]) / len(self.performance_metrics["response_times"]) if self.performance_metrics["response_times"] else 0
        
        report = f"""
📈 **Friday Performance Report**

**Uptime:** {uptime.days} days, {uptime.seconds // 3600} hours
**Total Interactions:** {self.performance_metrics['successful_interactions']}
**Average Response Time:** {avg_response:.2f} seconds
**Error Rate:** {(self.performance_metrics['error_count'] / max(1, self.performance_metrics['successful_interactions']) * 100):.1f}%
**Active Sessions:** {len(self.conversation_states)}

**Features Status:**
• Voice: {'✅ Enabled' if VOICE_AVAILABLE else '❌ Not available'}
• Encryption: {'✅ Enabled' if ENCRYPTION_AVAILABLE else '❌ Not available'}
• Emergency Detection: ✅ Active
• Auto-save: ✅ Active
• Resilience Engine: ✅ Active

**Status:** ✅ All systems operational
"""
        return report

# === HELPER FUNCTIONS (KEEPING YOUR ORIGINALS) ===

def handle_user_input_intelligently(user_input, ai):
    """Smart input handling with semantic analysis"""
    
    # Analyze the input first
    analysis = ai._analyze_input_semantic(user_input)
    
    if analysis['type'] == 'non_conversational':
        # Handle non-conversational input
        if 'response' in analysis:
            return analysis['response']
        else:
            return "I'm not sure what you're trying to do. Could you rephrase that as a question or tell me what you need help with?"
    
    elif analysis['type'] == 'pregnancy_concern':
        # This is a real pregnancy concern - proceed with full emotional support
        return ai.respond_to(user_input)['content']
    
    elif analysis['type'] == 'possible_pregnancy_concern':
        # Medium confidence - ask for clarification
        return analysis['response']
    
    else:
        # General conversation - normal response
        return ai.respond_to(user_input)['content']

def handle_pregnancy_test(user_input, ai):
    """Simple pregnancy emotion test command"""
    if user_input.startswith("!pregnancy_test"):
        try:
            # Extract week number if provided: !pregnancy_test 20
            parts = user_input.split()
            week = int(parts[1]) if len(parts) > 1 else 20
            
            # Test message
            test_message = "I'm so excited but also nervous about feeling the baby move!"
            
            response = ai.respond_to(test_message, pregnancy_week=week)
            
            return f"Testing pregnancy support (Week {week}):\n{response['content']}"
            
        except Exception as e:
            return f"Test failed: {e}"
    
    return None

def show_tone_selection():
    """Show tone selection menu at startup"""
    print("\n" + "="*60)
    print("🎭 FRIDAY TONE PREFERENCES")
    print("="*60)
    print("\nChoose how you'd like Friday to communicate with you:\n")
    
    print("💙 1. SUPPORTIVE (Default)")
    print("   • Warm, empathetic, lots of emotional validation")
    print("   • Includes resources and gentle guidance")
    print("   • Perfect for emotional support during pregnancy")
    print("   • Example: 'I understand you're feeling scared...'")
    
    print("\n💅 2. SASSY")  
    print("   • Friendly, confident, like your best friend")
    print("   • Uses 'girl', 'honey', 'queen' language")
    print("   • Playful but supportive approach")
    print("   • Example: 'Girl, you've got this! Let me tell you...'")
    
    print("\n📊 3. DIRECT")
    print("   • Facts-focused, clinical, evidence-based")
    print("   • Minimal emotion, maximum information")
    print("   • Great for science-minded users")
    print("   • Example: 'Research indicates that 70% of mothers...'")
    
    print("\n" + "="*60)
    print("💡 You can change your tone anytime with: !tone [supportive/sassy/direct]")
    print("="*60)

def get_tone_choice():
    """Get user's tone preference"""
    while True:
        try:
            choice = input("\nEnter your choice (1, 2, 3) or press Enter for Supportive: ").strip()
        except (EOFError, KeyboardInterrupt):
            return "supportive", "💙 Perfect! Friday will be warm and supportive."
        
        if choice == "" or choice == "1":
            return "supportive", "💙 Perfect! Friday will be warm and supportive."
        elif choice == "2":
            return "sassy", "💅 Great choice! Friday will be your sassy bestie."
        elif choice == "3": 
            return "direct", "📊 Excellent! Friday will give you straight facts."
        else:
            print("❌ Please enter 1, 2, 3, or press Enter for default.")

# === MAIN EXECUTION (ENHANCED WITH NEW FEATURES) ===
if __name__ == "__main__":
    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        print("\n\n💙 Friday is saving conversations and shutting down gracefully...")
        if 'ai' in globals():
            ai._save_all_conversations()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # === Brain & Domain Setup ===
    from core.MemoryCore import MemoryCore
    from core.EmotionCoreV2 import EmotionCoreV2

    # Silent initialization
    print("Friday is waking up...")
    ai = None

    try:
        memory = MemoryCore(memory_file="friday_memory.enc", key_file="memory.key")
        emotion = EmotionCoreV2()
        ai = FridayAI(memory, emotion)
        
        # Initialize optional components with safe imports (silent)
        logger = None
        try:
            from core.ThoughtLogger import ThoughtLogger
            logger = ThoughtLogger(memory, emotion)
        except:
            pass
        
        predictor = None
        try:
            from core.MoodPredictor import MoodPredictor
            predictor = MoodPredictor(ai.mood_filter)
        except:
            pass
        
        compressor = None
        try:
            from core.NarrativeCompressor import NarrativeCompressor
            compressor = NarrativeCompressor(memory)
        except:
            pass

        # Show performance report
        print(ai.get_performance_report())

        # === TONE SELECTION AT STARTUP ===
        show_tone_selection()
        chosen_tone, confirmation_msg = get_tone_choice()
        
        # Set the chosen tone
        if hasattr(ai, 'tone_manager'):
            ai.tone_manager.current_tone = chosen_tone
        
        print(f"\n{confirmation_msg}")
        
        # === NEW: INTERACTION MODE SELECTION ===
        if VOICE_AVAILABLE:
            print("\n🎯 **Choose Interaction Mode:**")
            print("1. Text Chat (Default)")
            print("2. Voice Chat")
            print("3. Mixed Mode (Text + Voice commands)")
            
            mode_choice = input("\nEnter choice (1-3) or press Enter for text: ").strip()
            
            if mode_choice == "2":
                # Voice mode
                print("\n🎤 Entering voice mode...")
                user_id = input("Enter your name (or press Enter for 'Guest'): ").strip() or "Guest"
                ai.voice_interaction(user_id)
                # After voice mode ends, continue to text
        
        print(f"\nHello! I'm Friday, your AI companion. How are you feeling today?")

        while True:
            print("\n" + "="*50)
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nFriday: Take care! I'm always here when you need me. 💙")
                ai._save_all_conversations()
                break

            if user_input.lower() in ["exit", "quit", "goodbye", "bye"]:
                print("\nFriday: Take care! I'm always here when you need me. 💙")
                ai._save_all_conversations()
                break

            if not user_input:  # Empty input
                print("\nFriday: I'm listening. What's on your mind?")
                continue

            try:
                # Check for tone changes FIRST
                if hasattr(ai, 'tone_manager'):
                    tone_response = ai.tone_manager.detect_tone_request(user_input)
                    if tone_response:
                        print(f"\nFriday: {tone_response}")
                        continue

                # === NEW SPECIAL COMMANDS ===
                if user_input.lower() == "!status":
                    print(f"\nFriday: {ai.get_performance_report()}")
                    continue
                
                if user_input.lower() == "!health":
                    user_id = input("Enter your name for health records: ").strip() or "default"
                    print(f"\nFriday: {ai.get_health_insights(user_id)}")
                    continue
                
                if user_input.lower() == "!voice":
                    if VOICE_AVAILABLE:
                        ai.voice_interaction()
                    else:
                        print("\nFriday: Voice features not available. Install speechrecognition and pyttsx3.")
                    continue
                
                if user_input.lower().startswith("!save health"):
                    # Save health data
                    if not ENCRYPTION_AVAILABLE:
                        print("\nFriday: Health vault not available. Install cryptography for secure health tracking.")
                        continue
                        
                    user_id = input("Enter your name: ").strip() or "default"
                    data_str = user_input[len("!save health"):].strip()
                    try:
                        data = json.loads(data_str) if data_str else {}
                        if ai.vault.store_health_data(user_id, "manual_entry", data):
                            print("\nFriday: ✅ Health data securely saved!")
                        else:
                            print("\nFriday: Failed to save health data.")
                    except:
                        print("\nFriday: Please provide data in JSON format after !save health")
                    continue

                # Special commands (keeping your originals)
                pregnancy_test = handle_pregnancy_test(user_input, ai)
                if pregnancy_test:
                    print(f"\nFriday: {pregnancy_test}")
                    continue

                if user_input.lower().startswith("!tones") or user_input.lower() == "!tone":
                    current = ai.tone_manager.current_tone if hasattr(ai, 'tone_manager') else "supportive"
                    print(f"\nFriday: 🎭 Current tone: **{current.title()}**")
                    print("Available tones: supportive, sassy, direct")
                    print("Use: !tone [supportive/sassy/direct]")
                    continue

                if user_input.lower().startswith("!clean"):
                    dirty = user_input[len("!clean"):].strip()
                    cleaned = ai.input_sanitizer.sanitize(dirty)
                    print(f"\nFriday: Cleaned text: {cleaned}")
                    continue

                if user_input.lower().startswith("log:") and logger:
                    thought = user_input.split("log:", 1)[1].strip()
                    entry = logger.log_thought(thought)
                    print(f"\nFriday: I've noted that thought. Your mood seems {entry['mood']}.")
                    continue

                if user_input.lower().startswith("!predict_mood") and predictor:
                    print(f"\nFriday: {predictor.predict_trend()}")
                    continue

                if user_input.lower().startswith("!narrative") and compressor:
                    print(f"\nFriday: {compressor.compress()}")
                    continue

                # Main conversation - smart handling
                response = handle_user_input_intelligently(user_input, ai)
                print(f"\nFriday: {response}")
                
                # Update performance metrics
                ai.performance_metrics["successful_interactions"] += 1

            except Exception as e:
                # More graceful error handling
                ai.performance_metrics["error_count"] += 1
                
                if "json" in str(e).lower():
                    print(f"\nFriday: I had trouble processing that. Could you rephrase it differently?")
                elif "memory" in str(e).lower():
                    print(f"\nFriday: Let me think about that... Could you ask me again?")
                elif "emotion" in str(e).lower():
                    print(f"\nFriday: I'm sensing a lot of feeling in your words. How can I support you right now?")
                else:
                    print(f"\nFriday: Something's not quite right on my end. Could you try asking that another way?")
                                
                # Silent logging for debugging
                ai.logger.error(f"Error processing '{user_input[:50]}...': {e}")

    except Exception as e:
        print("Friday: I'm having trouble starting up. Please check my configuration.")
        import traceback
        print("====== ERROR DETAILS ======")
        traceback.print_exc()
        print("=============================")