# =====================================
# File: FridayAI.py (Super Ultra Unstoppable Legendary Core)
# Purpose: Merges all features: Cognitive, Emotional, Pregnancy, Legendary & Unstoppable, Secure, Citation, and CLI/Voice
# =====================================

import logging
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
sys.stdout.reconfigure(encoding='utf-8')

logging.getLogger("transformers").setLevel(logging.CRITICAL)
logging.getLogger("sentence_transformers").setLevel(logging.CRITICAL)
logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger().setLevel(logging.CRITICAL)

from datetime import datetime, timedelta
import threading
import time
import re
import random
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dotenv import load_dotenv
from dataclasses import dataclass, field
from collections import defaultdict, deque

# === Optional Advanced Libraries ===
try:
    import speech_recognition as sr
    import pyttsx3
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

try:
    from cryptography.fernet import Fernet
    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False

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

import queue
import signal
import atexit
from pathlib import Path
import sqlite3
import warnings
warnings.filterwarnings("ignore")

# === Core Modules (Original Imports) ===
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
from core.MemoryContextInjector import MemoryContextInjector, inject
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

# === Pregnancy Enhancement (Optional) ===
try:
    from core.pregnancy.PregnancyEmotionCore import PregnancyEmotionCore
    PREGNANCY_EMOTION_AVAILABLE = True
except ImportError:
    PREGNANCY_EMOTION_AVAILABLE = False

# === Legacy/Non-core Modules ===
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

# Maternal Care Add-ons
from maternal_care import (
    SecureMaternalDatabase, 
    MaternalHealthProfile,
    OfflineCapableFriday,
    PrivacyTrustManager
)

# === BOOTSTRAP ===
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

# ====== LEGENDARY & UNSTOPPABLE ENHANCEMENT CLASSES ======

@dataclass
class LegendaryConversationMemory:
    """Enhanced conversation memory with context awareness"""
    def __init__(self, max_history=50):
        self.conversations = deque(maxlen=max_history)
        self.topic_memory = defaultdict(list)
        self.user_patterns = defaultdict(int)
    
    def add_exchange(self, user_input: str, ai_response: str, emotional_tone: str):
        timestamp = datetime.now()
        entry = {
            'user_input': user_input,
            'ai_response': ai_response,
            'emotional_tone': emotional_tone,
            'timestamp': timestamp,
            'topic_hash': self._get_topic_hash(user_input)
        }
        self.conversations.append(entry)
        topic = self._extract_topic(user_input)
        if topic:
            self.topic_memory[topic].append(entry)
        self.user_patterns[emotional_tone] += 1
    
    def find_similar_conversation(self, user_input: str, similarity_threshold=0.7) -> Optional[Dict]:
        current_topic_hash = self._get_topic_hash(user_input)
        current_keywords = set(self._extract_keywords(user_input))
        best_match = None
        best_score = 0
        for conv in reversed(list(self.conversations)):
            past_keywords = set(self._extract_keywords(conv['user_input']))
            if len(current_keywords) > 0 and len(past_keywords) > 0:
                intersection = current_keywords.intersection(past_keywords)
                union = current_keywords.union(past_keywords)
                jaccard_score = len(intersection) / len(union)
                if conv['emotional_tone'] in user_input.lower():
                    jaccard_score += 0.2
                if jaccard_score > similarity_threshold and jaccard_score > best_score:
                    best_match = conv
                    best_score = jaccard_score
        return best_match if best_match else None
    
    def _get_topic_hash(self, text: str) -> str:
        keywords = self._extract_keywords(text)
        topic_string = ' '.join(sorted(keywords[:3]))
        return hashlib.md5(topic_string.encode()).hexdigest()[:8]
    
    def _extract_topic(self, text: str) -> Optional[str]:
        pregnancy_topics = ['pregnancy', 'baby', 'birth', 'labor', 'prenatal', 'trimester']
        health_topics = ['anxiety', 'depression', 'stress', 'mood', 'sleep']
        text_lower = text.lower()
        for topic in pregnancy_topics + health_topics:
            if topic in text_lower:
                return topic
        return None
    
    def _extract_keywords(self, text: str) -> List[str]:
        stop_words = {'i', 'am', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = re.findall(r'\b\w+\b', text.lower())
        return [word for word in words if len(word) > 2 and word not in stop_words]

class GoalCoachingSystem:
    def __init__(self):
        self.active_goals = {}
        self.goal_templates = {
            'anxiety_management': {
                'title': 'Managing Pregnancy Anxiety',
                'steps': [
                    'Practice daily breathing exercises (5 minutes)',
                    'Keep a worry journal',
                    'Schedule regular check-ins with healthcare provider',
                    'Join a pregnancy support group'
                ],
                'check_in_days': 3
            },
            'birth_preparation': {
                'title': 'Birth Preparation Plan',
                'steps': [
                    'Research birthing options',
                    'Create birth plan document',
                    'Practice relaxation techniques',
                    'Pack hospital bag'
                ],
                'check_in_days': 7
            },
            'nutrition_health': {
                'title': 'Pregnancy Nutrition Goals',
                'steps': [
                    'Take prenatal vitamins daily',
                    'Eat 5 servings of fruits/vegetables daily',
                    'Stay hydrated (8+ glasses water)',
                    'Track weekly weight gain'
                ],
                'check_in_days': 5
            }
        }
        self.pending_check_ins = {}
    
    def detect_goal_opportunity(self, user_input: str, ai_response: str) -> Optional[str]:
        goal_triggers = {
            'anxiety_management': ['anxious', 'worried', 'scared', 'stress', 'overwhelmed'],
            'birth_preparation': ['birth', 'delivery', 'labor', 'due date', 'hospital'],
            'nutrition_health': ['eating', 'nutrition', 'weight', 'vitamins', 'healthy']
        }
        user_lower = user_input.lower()
        if any(word in ai_response.lower() for word in ['try', 'consider', 'help', 'suggest', 'recommend']):
            for goal_type, triggers in goal_triggers.items():
                if any(trigger in user_lower for trigger in triggers):
                    return goal_type
        return None
    
    def create_goal_offer(self, goal_type: str) -> str:
        template = self.goal_templates.get(goal_type)
        if not template:
            return ""
        offer = f"\n\n🎯 **Would you like me to help you create a personal plan?**\n"
        offer += f"I could help you work on: **{template['title']}**\n\n"
        offer += "This would include:\n"
        for i, step in enumerate(template['steps'][:3], 1):
            offer += f"• {step}\n"
        offer += f"\nI'd check in with you every {template['check_in_days']} days to see how you're doing.\n"
        offer += "**Interested? Just say 'yes' or 'create goal'!**"
        return offer
    
    def create_goal(self, goal_type: str, user_id: str = "default") -> str:
        template = self.goal_templates.get(goal_type)
        if not template:
            return "I couldn't create that goal. Let me know what you'd like help with!"
        goal_id = f"{goal_type}_{int(time.time())}"
        goal = {
            'id': goal_id,
            'type': goal_type,
            'title': template['title'],
            'steps': template['steps'].copy(),
            'completed_steps': [],
            'created_date': datetime.now(),
            'check_in_days': template['check_in_days'],
            'last_check_in': datetime.now()
        }
        if user_id not in self.active_goals:
            self.active_goals[user_id] = []
        self.active_goals[user_id].append(goal)
        next_check = datetime.now() + timedelta(days=template['check_in_days'])
        self.pending_check_ins[goal_id] = next_check
        response = f"🎯 **Goal Created: {template['title']}**\n\n"
        response += "Your action steps:\n"
        for i, step in enumerate(template['steps'], 1):
            response += f"{i}. {step}\n"
        response += f"\n💙 I'll check in with you in {template['check_in_days']} days to see how you're progressing!"
        return response
    
    def check_for_due_check_ins(self, user_id: str = "default") -> Optional[str]:
        now = datetime.now()
        due_check_ins = []
        for goal_id, check_date in self.pending_check_ins.items():
            if now >= check_date:
                for goal in self.active_goals.get(user_id, []):
                    if goal['id'] == goal_id:
                        due_check_ins.append(goal)
                        break
        if due_check_ins:
            goal = due_check_ins[0]
            return self._create_check_in_message(goal)
        return None
    
    def _create_check_in_message(self, goal: Dict) -> str:
        days_since = (datetime.now() - goal['last_check_in']).days
        message = f"🎯 **Goal Check-in: {goal['title']}**\n\n"
        message += f"It's been {days_since} days since we set up your goal.\n"
        message += "How are you doing with:\n\n"
        for i, step in enumerate(goal['steps'], 1):
            status = "✅" if step in goal['completed_steps'] else "📋"
            message += f"{status} {i}. {step}\n"
        message += "\n💬 **How's it going? Any challenges or wins to share?**"
        return message

class RichOutputFormatter:
    def __init__(self):
        self.colors = {
            'empathy': '\033[96m',      # Cyan
            'success': '\033[92m',      # Green
            'warning': '\033[93m',      # Yellow
            'error': '\033[91m',        # Red
            'info': '\033[94m',         # Blue
            'bold': '\033[1m',          # Bold
            'end': '\033[0m'            # End formatting
        }
        self.use_colors = self._supports_color()
    def _supports_color(self) -> bool:
        return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty() and os.name != 'nt'
    def format_response(self, text: str, response_type: str = 'normal') -> str:
        if not self.use_colors:
            return self._format_plain(text)
        if response_type == 'empathy':
            return self._format_empathy(text)
        elif response_type == 'goal':
            return self._format_goal(text)
        elif response_type == 'resource':
            return self._format_resource(text)
        else:
            return self._format_normal(text)
    def _format_empathy(self, text: str) -> str:
        text = re.sub(r'(I understand|I hear you|That sounds|It\'s normal)', 
                     f"{self.colors['empathy']}\\1{self.colors['end']}", text)
        text = re.sub(r'(\*\*.*?\*\*)', 
                     f"{self.colors['bold']}\\1{self.colors['end']}", text)
        return text
    def _format_goal(self, text: str) -> str:
        text = re.sub(r'🎯 \*\*(.*?)\*\*', 
                     f"🎯 {self.colors['bold']}{self.colors['success']}\\1{self.colors['end']}", text)
        text = text.replace('✅', f"{self.colors['success']}✅{self.colors['end']}")
        return text
    def _format_resource(self, text: str) -> str:
        text = re.sub(r'(\*\*📚 .*?\*\*|\*\*📱 .*?\*\*|\*\*🤝 .*?\*\*)', 
                     f"{self.colors['info']}\\1{self.colors['end']}", text)
        return text
    def _format_normal(self, text: str) -> str:
        text = re.sub(r'\*\*(.*?)\*\*', 
                     f"{self.colors['bold']}\\1{self.colors['end']}", text)
        return text
    def _format_plain(self, text: str) -> str:
        text = re.sub(r'\*\*(.*?)\*\*', r'[\1]', text)
        return text

class SelfEvaluationSystem:
    def __init__(self):
        self.interaction_count = 0
        self.feedback_requests = 0
        self.user_feedback_history = []
        self.tone_adjustments = defaultdict(int)
        self.last_feedback_request = None
    def should_request_feedback(self) -> bool:
        self.interaction_count += 1
        if self.last_feedback_request:
            time_since_last = datetime.now() - self.last_feedback_request
            if time_since_last.total_seconds() < 600:
                return False
        if self.interaction_count in [7, 20, 50] or (self.interaction_count > 50 and self.interaction_count % 25 == 0):
            return True
        if self.interaction_count > 10 and random.random() < 0.08:
            return True
        return False
    def generate_feedback_request(self, current_tone: str, recent_topics: List[str]) -> str:
        self.feedback_requests += 1
        self.last_feedback_request = datetime.now()
        requests = [
            f"💭 Quick check-in: How am I doing with my {current_tone} tone? Should I adjust anything?",
            f"🎯 I want to make sure I'm helping you well. How's our conversation style working for you?",
            f"💙 Am I being too {current_tone}, or would you prefer a different approach?",
            f"🔄 We've covered {', '.join(recent_topics[:2])} today. How can I better support you?"
        ]
        return random.choice(requests)
    def process_feedback(self, feedback: str, current_tone: str) -> str:
        feedback_lower = feedback.lower()
        self.user_feedback_history.append({
            'feedback': feedback,
            'timestamp': datetime.now(),
            'context_tone': current_tone
        })
        if any(word in feedback_lower for word in ['good', 'great', 'perfect', 'love', 'helpful']):
            return "💙 Thank you! I'm glad I'm helping. I'll keep doing what's working!"
        elif any(word in feedback_lower for word in ['more supportive', 'gentler', 'softer']):
            self.tone_adjustments['more_supportive'] += 1
            return "💙 I'll be more gentle and supportive. Thank you for letting me know."
        elif any(word in feedback_lower for word in ['less', 'too much', 'direct', 'factual']):
            self.tone_adjustments['less_emotional'] += 1
            return "📊 Got it! I'll be more direct and focus on facts. Thanks for the guidance."
        elif any(word in feedback_lower for word in ['sassy', 'fun', 'casual', 'friend']):
            self.tone_adjustments['more_casual'] += 1
            return "💅 Perfect! I'll bring more personality and sass to our chats!"
        else:
            return "💭 Thanks for the feedback! I'm always learning how to better support you."

class CitationSystem:
    def __init__(self):
        self.medical_sources = {
            'pregnancy_facts': {
                'url': 'https://www.who.int/news-room/fact-sheets/detail/pregnancy',
                'title': 'WHO Pregnancy Facts',
                'domain': 'pregnancy'
            },
            'maternal_health': {
                'url': 'https://www.cdc.gov/reproductivehealth/maternalinfanthealth/',
                'title': 'CDC Maternal Health',
                'domain': 'health'
            },
            'prenatal_care': {
                'url': 'https://www.acog.org/womens-health/faqs/prenatal-care',
                'title': 'ACOG Prenatal Care Guidelines',
                'domain': 'medical'
            }
        }
        self.citation_patterns = {
            'research shows': 'medical_research',
            'studies indicate': 'medical_research',
            'doctors recommend': 'medical_advice',
            'healthcare providers': 'medical_advice',
            'according to experts': 'expert_opinion'
        }
    def add_citations(self, response: str, topic_context: str) -> str:
        response_lower = response.lower()
        for pattern, citation_type in self.citation_patterns.items():
            if pattern in response_lower:
                citation = self._get_relevant_citation(topic_context, citation_type)
                if citation:
                    response += f"\n\n📚 **Source:** {citation}"
                    break
        return response
    def _get_relevant_citation(self, topic: str, citation_type: str) -> Optional[str]:
        topic_lower = topic.lower()
        for source_key, source_info in self.medical_sources.items():
            if any(keyword in topic_lower for keyword in source_info['domain'].split()):
                return f"[{source_info['title']}]({source_info['url']})"
        if citation_type in ['medical_advice', 'medical_research']:
            return "[American College of Obstetricians and Gynecologists](https://www.acog.org)"
        return None

class KnowledgeInjectionSystem:
    def __init__(self):
        self.pregnancy_facts = [
            "A baby's heart starts beating around 6 weeks of pregnancy.",
            "Pregnant women's blood volume increases by 30-50% during pregnancy.",
            "The baby can hear sounds from outside the womb starting around 20 weeks.",
            "Morning sickness affects about 70-80% of pregnant women.",
            "A baby's fingerprints are formed by 18 weeks of pregnancy.",
            "The sense of smell often becomes stronger during pregnancy due to hormonal changes.",
            "Babies can taste what their mothers eat through the amniotic fluid.",
            "The uterus grows from the size of a pear to the size of a watermelon during pregnancy."
        ]
        self.wellness_facts = [
            "Deep breathing for just 5 minutes can significantly reduce stress hormones.",
            "Prenatal yoga can help reduce anxiety and improve sleep quality.",
            "Talking to your baby in the womb can help with bonding and brain development.",
            "Keeping a gratitude journal during pregnancy is linked to better emotional well-being.",
            "Light exercise during pregnancy can reduce labor time and complications."
        ]
        self.last_fact_time = None
        self.used_facts = set()
    def should_add_fact(self) -> bool:
        if self.last_fact_time:
            time_since = datetime.now() - self.last_fact_time
            if time_since.total_seconds() < 180:
                return False
        return random.random() < 0.15
    def get_relevant_fact(self, topic_context: str, emotional_tone: str) -> str:
        self.last_fact_time = datetime.now()
        if any(word in topic_context.lower() for word in ['stress', 'anxiety', 'worried', 'overwhelmed']):
            facts_pool = self.wellness_facts
        else:
            facts_pool = self.pregnancy_facts
        available_facts = [f for f in facts_pool if f not in self.used_facts]
        if not available_facts:
            self.used_facts.clear()
            available_facts = facts_pool
        fact = random.choice(available_facts)
        self.used_facts.add(fact)
        return f"\n\n💡 **Did you know?** {fact}"

# === Unstoppable Features ===

class ResilienceEngine:
    def __init__(self):
        self.error_history = deque(maxlen=1000)
        self.recovery_strategies = {}
        self.circuit_breakers = {}
        self.fallback_responses = self._load_fallbacks()
    def _load_fallbacks(self):
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
        from functools import wraps
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
                        time.sleep(0.1 * attempt)
                        continue
                    else:
                        return self._get_fallback_response(func.__name__, e)
        return wrapper
    def _get_fallback_response(self, func_name: str, error: Exception):
        if "memory" in func_name.lower():
            responses = self.fallback_responses["memory_error"]
        elif "emotion" in func_name.lower():
            responses = self.fallback_responses["emotional_overload"]
        else:
            responses = self.fallback_responses["general_error"]
        return random.choice(responses)

class PredictiveAnalytics:
    def __init__(self):
        self.milestone_predictor = self._init_milestone_model()
        self.mood_predictor = self._init_mood_model()
        self.need_predictor = self._init_need_model()
    def _init_milestone_model(self):
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
        predictions = []
        for week, milestones in self.milestone_predictor.items():
            if current_week <= week <= current_week + 4:
                for milestone in milestones:
                    predictions.append(f"Week {week}: {milestone}")
        return predictions
    def predict_emotional_needs(self, mood_history: deque) -> Dict[str, float]:
        if not mood_history:
            return {"support": 0.5, "information": 0.3, "reassurance": 0.2}
        recent_moods = list(mood_history)[-10:]
        mood_counts = defaultdict(int)
        for mood in recent_moods:
            mood_counts[mood] += 1
        if mood_counts.get("anxious", 0) > 3:
            return {"reassurance": 0.7, "calming": 0.2, "information": 0.1}
        elif mood_counts.get("sad", 0) > 2:
            return {"empathy": 0.6, "support": 0.3, "positivity": 0.1}
        else:
            return {"information": 0.5, "support": 0.3, "encouragement": 0.2}

class VoiceInterface:
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
        voices = self.engine.getProperty('voices')
        for voice in voices:
            if "female" in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                break
        self.engine.setProperty('rate', 180)
        self.engine.setProperty('volume', 0.9)
    def listen(self, timeout=5):
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
        if not self.enabled:
            return
        clean_text = re.sub(r'[*_#`]', '', text)
        clean_text = re.sub(r'\n+', '. ', clean_text)
        try:
            self.engine.say(clean_text)
            self.engine.runAndWait()
        except:
            pass

class SecureDataVault:
    def __init__(self, vault_path="maternal_vault.db"):
        self.vault_path = vault_path
        self.enabled = ENCRYPTION_AVAILABLE
        if self.enabled:
            self.cipher_suite = Fernet(self._get_or_create_key())
            self._init_vault()
    def _get_or_create_key(self):
        key_path = Path("vault.key")
        if key_path.exists():
            return key_path.read_bytes()
        else:
            key = Fernet.generate_key()
            key_path.write_bytes(key)
            return key
    def _init_vault(self):
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
        if not self.enabled:
            return False
        try:
            data_bytes = json.dumps(data).encode()
            encrypted = self.cipher_suite.encrypt(data_bytes)
            checksum = hashlib.sha256(data_bytes).hexdigest()
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
    def __init__(self):
        self.emergency_keywords = [
            "bleeding", "severe pain", "can't breathe", "contractions",
            "water broke", "baby not moving", "dizzy", "faint",
            "chest pain", "severe headache", "vision problems"
        ]
        self.emergency_contacts = []
        self.location_service = None
    def check_emergency(self, user_input: str) -> Tuple[bool, str]:
        input_lower = user_input.lower()
        for keyword in self.emergency_keywords:
            if keyword in input_lower:
                return True, keyword
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

# === TONE SYSTEM ===
class SimpleToneManager:
    def __init__(self):
        self.current_tone = "supportive"
    def detect_tone_request(self, user_input):
        input_lower = user_input.lower().strip()
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
        if self.current_tone == "sassy":
            return self._make_sassy(original_response)
        elif self.current_tone == "direct":
            return self._make_direct(original_response)
        else:
            return original_response
    def _make_sassy(self, text):
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
        prefix = random.choice(sassy_prefixes)
        ending = random.choice(sassy_endings)
        modified = text.replace("It's important to", "Girl, you NEED to")
        modified = modified.replace("You should", "Honey, you better")
        modified = modified.replace("Healthcare providers", "Your doc (who went to school forever)")
        return f"{prefix}\n\n{modified}\n\n{ending}"
    def _make_direct(self, text):
        direct_prefixes = [
            "Based on medical evidence:",
            "Clinical facts:",
            "Key information:"
        ]
        direct_endings = [
            "Consult your healthcare provider for personalized advice.",
            "This is based on current medical evidence."
        ]
        prefix = random.choice(direct_prefixes)
        ending = random.choice(direct_endings)
        modified = text.replace("I understand", "Research indicates")
        modified = modified.replace("I'm here for you", "Support is available")
        modified = modified.replace("Don't worry", "Evidence suggests")
        return f"{prefix}\n\n{modified}\n\n{ending}"

# ====== FRIDAYAI MASTER CLASS ======
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
        if PREGNANCY_EMOTION_AVAILABLE:
            self.pregnancy_emotion = PregnancyEmotionCore()
        else:
            self.pregnancy_emotion = None
        # Legendary & unstoppable features
        self.legendary_memory = LegendaryConversationMemory()
        self.goal_coach = GoalCoachingSystem()
        self.output_formatter = RichOutputFormatter()
        self.self_eval = SelfEvaluationSystem()
        self.citation_system = CitationSystem()
        self.knowledge_injection = KnowledgeInjectionSystem()
        self.resilience = ResilienceEngine()
        self.predictive = PredictiveAnalytics()
        self.voice = VoiceInterface()
        self.vault = SecureDataVault()
        self.emergency = EmergencyProtocol()
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
        self.empathy_responses = self._load_empathy_safe()
        self.tone_rewriter = ToneRewriterCore()
        # Any further initialization...
    def _configure_logging(self):
        self.logger = logging.getLogger("FridayAI")
        self.logger.setLevel(logging.CRITICAL)
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
        # Fallback in case empathy file not found
        return [
            "I'm here for you. 💙",
            "That sounds really tough, but I'm with you.",
            "You matter and your feelings are valid.",
            "I'm listening—take your time."
        ]
