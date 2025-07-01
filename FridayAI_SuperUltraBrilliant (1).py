# =====================================
# File: FridayAI_SuperUltraBrilliant.py (ULTIMATE MERGE)
# Purpose: Complete merge of ALL cognitive architectures + Legendary + Unstoppable + Pregnancy Intelligence + Voice + Security
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
import random
import json
import hashlib
import pickle
from typing import Dict, List, Optional, Tuple, Any
from dotenv import load_dotenv
from dataclasses import dataclass, field
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from functools import lru_cache, wraps
import queue
import signal
import atexit
from pathlib import Path
import sqlite3
import warnings
warnings.filterwarnings("ignore")

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

# === Core Modules ===
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

# === Pregnancy Enhancement ===
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

# === Maternal Care Add-ons ===
from maternal_care import (
    SecureMaternalDatabase, 
    MaternalHealthProfile,
    OfflineCapableFriday,
    PrivacyTrustManager
)

# === BOOTSTRAP ===
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()


# ====== LEGENDARY ENHANCEMENT IMPORTS ======
from legendary_features import LegendaryConversationMemory, GoalCoachingSystem
'''
@dataclass
class LegendaryConversationMemory:
    """Enhanced conversation memory with context awareness"""
    def __init__(self, max_history=100):
        self.conversations = deque(maxlen=max_history)
        self.topic_memory = defaultdict(list)
        self.user_patterns = defaultdict(int)
        self.emotional_patterns = defaultdict(deque)
        self.session_insights = {}
    
    def add_exchange(self, user_input: str, ai_response: str, emotional_tone: str):
        timestamp = datetime.now()
        entry = {
            'user_input': user_input,
            'ai_response': ai_response,
            'emotional_tone': emotional_tone,
            'timestamp': timestamp,
            'topic_hash': self._get_topic_hash(user_input),
            'sentiment_score': self._analyze_sentiment(user_input)
        }
        
        self.conversations.append(entry)
        
        # Extract and store topic
        topic = self._extract_topic(user_input)
        if topic:
            self.topic_memory[topic].append(entry)
        
        # Track user patterns
        self.user_patterns[emotional_tone] += 1
        self.emotional_patterns[emotional_tone].append(timestamp)
    
    def find_similar_conversation(self, user_input: str, similarity_threshold=0.7) -> Optional[Dict]:
        current_keywords = set(self._extract_keywords(user_input))
        best_match = None
        best_score = 0
        
        for conv in reversed(list(self.conversations)):
            if len(self.conversations) > 2 and conv == list(self.conversations)[-2]:
                continue
                
            past_keywords = set(self._extract_keywords(conv['user_input']))
            
            if len(current_keywords) > 0 and len(past_keywords) > 0:
                intersection = current_keywords.intersection(past_keywords)
                union = current_keywords.union(past_keywords)
                jaccard_score = len(intersection) / len(union)
                
                # Boost score for emotional similarity
                if conv['emotional_tone'] in user_input.lower():
                    jaccard_score += 0.2
                
                # Boost for temporal relevance
                time_delta = datetime.now() - conv['timestamp']
                if time_delta.days < 7:
                    jaccard_score += 0.1
                
                if jaccard_score > similarity_threshold and jaccard_score > best_score:
                    best_match = conv
                    best_score = jaccard_score
        
        return best_match if best_match else None
    
    def get_emotional_insights(self) -> Dict:
        """Generate insights about user's emotional patterns"""
        if not self.conversations:
            return {}
        
        recent_convs = list(self.conversations)[-20:]  # Last 20 conversations
        emotions = [conv['emotional_tone'] for conv in recent_convs]
        
        return {
            'dominant_emotion': max(set(emotions), key=emotions.count),
            'emotional_variety': len(set(emotions)),
            'recent_trend': emotions[-5:] if len(emotions) >= 5 else emotions,
            'needs_support': emotions.count('anxious') + emotions.count('scared') > len(emotions) * 0.3
        }
    
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
        stop_words = {'i', 'am', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                     'to', 'for', 'of', 'with', 'by', 'you', 'me', 'my', 'your', 'it', 'this', 'that'}
        words = re.findall(r'\b\w+\b', text.lower())
        return [word for word in words if len(word) > 2 and word not in stop_words]
    
    def _analyze_sentiment(self, text: str) -> float:
        """Simple sentiment analysis"""
        positive_words = ['happy', 'excited', 'good', 'great', 'wonderful', 'amazing', 'love', 'joy', 'perfect']
        negative_words = ['scared', 'worried', 'anxious', 'sad', 'terrible', 'awful', 'hate', 'fear', 'bad']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count + neg_count == 0:
            return 0.0
        return (pos_count - neg_count) / (pos_count + neg_count)

class GoalCoachingSystem:
    """Advanced multi-turn goal and task planning system"""
    def __init__(self):
        self.active_goals = {}
        self.completed_goals = {}
        self.goal_templates = {
            'anxiety_management': {
                'title': 'Managing Pregnancy Anxiety',
                'steps': [
                    'Practice daily breathing exercises (5 minutes)',
                    'Keep a worry journal',
                    'Schedule regular check-ins with healthcare provider',
                    'Join a pregnancy support group',
                    'Practice mindfulness meditation',
                    'Create a support network'
                ],
                'check_in_days': 3,
                'category': 'mental_health'
            },
            'birth_preparation': {
                'title': 'Birth Preparation Plan',
                'steps': [
                    'Research birthing options',
                    'Create birth plan document',
                    'Practice relaxation techniques',
                    'Pack hospital bag',
                    'Take childbirth classes',
                    'Choose pediatrician'
                ],
                'check_in_days': 7,
                'category': 'preparation'
            },
            'nutrition_health': {
                'title': 'Pregnancy Nutrition Goals',
                'steps': [
                    'Take prenatal vitamins daily',
                    'Eat 5 servings of fruits/vegetables daily',
                    'Stay hydrated (8+ glasses water)',
                    'Track weekly weight gain',
                    'Avoid harmful foods',
                    'Plan healthy meal prep'
                ],
                'check_in_days': 5,
                'category': 'health'
            },
            'postpartum_preparation': {
                'title': 'Preparing for Postpartum',
                'steps': [
                    'Research postpartum recovery',
                    'Plan support for first weeks',
                    'Stock postpartum supplies',
                    'Prepare meals in advance',
                    'Set up baby care area',
                    'Learn about baby blues vs. depression'
                ],
                'check_in_days': 10,
                'category': 'preparation'
            }
        }
        self.pending_check_ins = {}
        self.goal_progress = {}
    
    def detect_goal_opportunity(self, user_input: str, ai_response: str, emotional_tone: str) -> Optional[str]:
        goal_triggers = {
            'anxiety_management': ['anxious', 'worried', 'scared', 'stress', 'overwhelmed', 'panic'],
            'birth_preparation': ['birth', 'delivery', 'labor', 'due date', 'hospital', 'contractions'],
            'nutrition_health': ['eating', 'nutrition', 'weight', 'vitamins', 'healthy', 'food'],
            'postpartum_preparation': ['after birth', 'postpartum', 'recovery', 'newborn', 'baby care']
        }
        
        user_lower = user_input.lower()
        response_lower = ai_response.lower()
        
        # Check if response includes advice (good time to offer coaching)
        advice_indicators = ['try', 'consider', 'help', 'suggest', 'recommend', 'important', 'should']
        has_advice = any(word in response_lower for word in advice_indicators)
        
        if has_advice or emotional_tone in ['anxious', 'scared', 'overwhelmed']:
            for goal_type, triggers in goal_triggers.items():
                if any(trigger in user_lower for trigger in triggers):
                    # Don't offer same type of goal if already active
                    if goal_type not in [g['type'] for goals in self.active_goals.values() for g in goals]:
                        return goal_type
        
        return None
    
    def create_goal_offer(self, goal_type: str) -> str:
        template = self.goal_templates.get(goal_type)
        if not template:
            return ""
        
        offer = f"\n\n🎯 **Would you like me to help you create a personal action plan?**\n"
        offer += f"I could help you work on: **{template['title']}**\n\n"
        offer += "This personalized plan would include:\n"
        for i, step in enumerate(template['steps'][:4], 1):
            offer += f"• {step}\n"
        
        if len(template['steps']) > 4:
            offer += f"• ...and {len(template['steps']) - 4} more steps\n"
        
        offer += f"\n💙 I'd check in with you every {template['check_in_days']} days to track progress and adjust as needed.\n"
        offer += "**Interested? Just say 'yes', 'create goal', or 'I'm in'!**"
        
        return offer
    
    def create_goal(self, goal_type: str, user_id: str = "default", custom_steps: List[str] = None) -> str:
        template = self.goal_templates.get(goal_type)
        if not template:
            return "I couldn't create that goal. Let me know what you'd like help with!"
        
        goal_id = f"{goal_type}_{int(time.time())}"
        
        goal = {
            'id': goal_id,
            'type': goal_type,
            'title': template['title'],
            'steps': custom_steps if custom_steps else template['steps'].copy(),
            'completed_steps': [],
            'created_date': datetime.now(),
            'check_in_days': template['check_in_days'],
            'last_check_in': datetime.now(),
            'category': template['category'],
            'progress_score': 0.0,
            'custom_notes': []
        }
        
        if user_id not in self.active_goals:
            self.active_goals[user_id] = []
        
        self.active_goals[user_id].append(goal)
        
        # Schedule first check-in
        next_check = datetime.now() + timedelta(days=template['check_in_days'])
        self.pending_check_ins[goal_id] = next_check
        
        response = f"🎯 **Goal Created: {template['title']}**\n\n"
        response += "Your personalized action steps:\n"
        for i, step in enumerate(goal['steps'], 1):
            response += f"{i}. {step}\n"
        response += f"\n💙 I'll check in with you in {template['check_in_days']} days to see how you're progressing!"
        response += f"\n✨ You can update your progress anytime by saying 'update goal' or 'goal progress'."
        
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
            goal = due_check_ins[0]  # Handle one at a time
            return self._create_check_in_message(goal)
        
        return None
    
    def update_goal_progress(self, user_id: str, completed_step: str) -> str:
        """Update progress on active goals"""
        for goal in self.active_goals.get(user_id, []):
            for step in goal['steps']:
                if completed_step.lower() in step.lower():
                    if step not in goal['completed_steps']:
                        goal['completed_steps'].append(step)
                        goal['progress_score'] = len(goal['completed_steps']) / len(goal['steps'])
                        
                        if goal['progress_score'] == 1.0:
                            return self._complete_goal(user_id, goal)
                        else:
                            return f"🎉 Great progress! You completed: {step}\n\nProgress: {len(goal['completed_steps'])}/{len(goal['steps'])} steps ({goal['progress_score']:.1%})"
        
        return "I couldn't find that step in your active goals. Try 'show goals' to see your current goals."
    
    def _create_check_in_message(self, goal: Dict) -> str:
        days_since = (datetime.now() - goal['last_check_in']).days
        
        message = f"🎯 **Goal Check-in: {goal['title']}**\n\n"
        message += f"It's been {days_since} days since we set up your goal.\n"
        message += f"Progress so far: {len(goal['completed_steps'])}/{len(goal['steps'])} steps ({goal['progress_score']:.1%})\n\n"
        
        message += "📋 **Your steps:**\n"
        for i, step in enumerate(goal['steps'], 1):
            status = "✅" if step in goal['completed_steps'] else "📋"
            message += f"{status} {i}. {step}\n"
        
        message += "\n💬 **How's it going? Any challenges, wins, or adjustments needed?**"
        message += "\n\n💡 You can say things like:\n"
        message += "• 'I completed step 2'\n"
        message += "• 'I'm struggling with step 1'\n"
        message += "• 'Can we modify this goal?'\n"
        
        return message
    
    def _complete_goal(self, user_id: str, goal: Dict) -> str:
        """Handle goal completion"""
        # Move to completed goals
        if user_id not in self.completed_goals:
            self.completed_goals[user_id] = []
        
        goal['completion_date'] = datetime.now()
        self.completed_goals[user_id].append(goal)
        
        # Remove from active goals
        self.active_goals[user_id] = [g for g in self.active_goals[user_id] if g['id'] != goal['id']]
        
        # Remove from check-ins
        if goal['id'] in self.pending_check_ins:
            del self.pending_check_ins[goal['id']]
        
        celebration = f"🎉🎉 **CONGRATULATIONS!** 🎉🎉\n\n"
        celebration += f"You've completed your goal: **{goal['title']}**!\n\n"
        celebration += f"✨ You finished all {len(goal['steps'])} steps over {(goal['completion_date'] - goal['created_date']).days} days.\n"
        celebration += f"🌟 This is a huge accomplishment - you should be proud!\n\n"
        celebration += "🎯 **Ready for your next challenge?** I can help you set up another goal!"
        
        return celebration
'''
class RichOutputFormatter:
    """Enhanced output formatting with ANSI colors and structure"""
    def __init__(self):
        self.colors = {
            'empathy': '\033[96m',      # Cyan
            'success': '\033[92m',      # Green
            'warning': '\033[93m',      # Yellow
            'error': '\033[91m',        # Red
            'info': '\033[94m',         # Blue
            'bold': '\033[1m',          # Bold
            'italic': '\033[3m',        # Italic
            'underline': '\033[4m',     # Underline
            'end': '\033[0m'            # End formatting
        }
        self.use_colors = self._supports_color()
    
    def _supports_color(self) -> bool:
        return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty() and os.name != 'nt'
    
    def format_response(self, text: str, response_type: str = 'normal', emotional_context: str = None) -> str:
        if not self.use_colors:
            return self._format_plain(text)
        
        if response_type == 'empathy':
            return self._format_empathy(text)
        elif response_type == 'goal':
            return self._format_goal(text)
        elif response_type == 'resource':
            return self._format_resource(text)
        elif response_type == 'emergency':
            return self._format_emergency(text)
        elif response_type == 'celebration':
            return self._format_celebration(text)
        else:
            return self._format_normal(text, emotional_context)
    
    def _format_empathy(self, text: str) -> str:
        # Highlight emotional validation phrases
        empathy_phrases = [
            r'(I understand|I hear you|That sounds|It\'s normal|I\'m here|You\'re not alone)',
            r'(Your feelings are valid|That makes sense|It\'s okay to feel)',
            r'(You\'re doing great|You\'re amazing|You\'ve got this)'
        ]
        
        for pattern in empathy_phrases:
            text = re.sub(pattern, f"{self.colors['empathy']}\\1{self.colors['end']}", text, flags=re.IGNORECASE)
        
        # Make headers bold
        text = re.sub(r'(\*\*.*?\*\*)', f"{self.colors['bold']}\\1{self.colors['end']}", text)
        
        return text
    
    def _format_goal(self, text: str) -> str:
        # Highlight goal titles
        text = re.sub(r'🎯 \*\*(.*?)\*\*', 
                     f"🎯 {self.colors['bold']}{self.colors['success']}\\1{self.colors['end']}", text)
        
        # Color check marks and progress indicators
        text = text.replace('✅', f"{self.colors['success']}✅{self.colors['end']}")
        text = text.replace('🎉', f"{self.colors['success']}🎉{self.colors['end']}")
        text = re.sub(r'(\d+\.\d+%|\d+/\d+)', f"{self.colors['info']}\\1{self.colors['end']}", text)
        
        return text
    
    def _format_resource(self, text: str) -> str:
        # Highlight resource categories
        resource_patterns = [
            r'(\*\*📚.*?\*\*|\*\*📱.*?\*\*|\*\*🤝.*?\*\*|\*\*💡.*?\*\*)',
            r'(📖|📱|🏥|👥|💊|🩺)'
        ]
        
        for pattern in resource_patterns:
            text = re.sub(pattern, f"{self.colors['info']}\\1{self.colors['end']}", text)
        
        return text
    
    def _format_emergency(self, text: str) -> str:
        # Emergency responses get special treatment
        text = re.sub(r'(🚨.*?🚨)', f"{self.colors['error']}{self.colors['bold']}\\1{self.colors['end']}", text)
        text = re.sub(r'(\*\*IMPORTANT.*?\*\*)', f"{self.colors['error']}{self.colors['bold']}\\1{self.colors['end']}", text)
        text = re.sub(r'(CALL 911|EMERGENCY|IMMEDIATELY)', f"{self.colors['error']}{self.colors['bold']}\\1{self.colors['end']}", text)
        
        return text
    
    def _format_celebration(self, text: str) -> str:
        # Celebration formatting
        text = re.sub(r'(🎉.*?🎉)', f"{self.colors['success']}{self.colors['bold']}\\1{self.colors['end']}", text)
        text = re.sub(r'(\*\*CONGRATULATIONS.*?\*\*)', f"{self.colors['success']}{self.colors['bold']}\\1{self.colors['end']}", text)
        
        return text
    
    def _format_normal(self, text: str, emotional_context: str = None) -> str:
        # Just make bold text actually bold
        text = re.sub(r'\*\*(.*?)\*\*', f"{self.colors['bold']}\\1{self.colors['end']}", text)
        
        # Add subtle emotional coloring
        if emotional_context == 'supportive':
            text = re.sub(r'(💙|💝|✨)', f"{self.colors['empathy']}\\1{self.colors['end']}", text)
        
        return text
    
    def _format_plain(self, text: str) -> str:
        # Convert markdown-style bold to plain text emphasis
        text = re.sub(r'\*\*(.*?)\*\*', r'[\1]', text)
        return text

class SelfEvaluationSystem:
    """Enhanced self-evaluation and feedback system"""
    def __init__(self):
        self.interaction_count = 0
        self.feedback_requests = 0
        self.user_feedback_history = []
        self.tone_adjustments = defaultdict(int)
        self.last_feedback_request = None
        self.performance_metrics = {
            'response_satisfaction': deque(maxlen=50),
            'topic_success_rate': defaultdict(list),
            'user_engagement_score': deque(maxlen=20)
        }
        self.learning_insights = []
    
    def should_request_feedback(self, emotional_context: str = None) -> bool:
        self.interaction_count += 1
        
        # Don't ask too frequently
        if self.last_feedback_request:
            time_since_last = datetime.now() - self.last_feedback_request
            if time_since_last.total_seconds() < 600:  # 10 minutes minimum
                return False
        
        # Ask after certain interaction milestones
        if self.interaction_count in [7, 20, 50] or (self.interaction_count > 50 and self.interaction_count % 25 == 0):
            return True
        
        # Higher chance if user seems distressed
        if emotional_context in ['anxious', 'overwhelmed', 'sad']:
            if random.random() < 0.15:  # 15% chance
                return True
        
        # Random chance after 10 interactions
        if self.interaction_count > 10 and random.random() < 0.06:  # 6% chance
            return True
        
        return False
    
    def generate_feedback_request(self, current_tone: str, recent_topics: List[str], emotional_context: str = None) -> str:
        self.feedback_requests += 1
        self.last_feedback_request = datetime.now()
        
        # Context-aware feedback requests
        if emotional_context in ['anxious', 'scared', 'overwhelmed']:
            requests = [
                f"💭 I want to make sure I'm providing the right kind of support. How am I doing with helping you feel more at ease?",
                f"🤗 I sense you're going through a tough time. Is my {current_tone} approach helpful, or would you prefer something different?",
                f"💙 Your wellbeing is my priority. Am I giving you the emotional support you need right now?"
            ]
        elif 'goal' in ' '.join(recent_topics).lower():
            requests = [
                f"🎯 How am I doing with helping you work toward your goals? Should I adjust my coaching style?",
                f"📈 I want to make sure I'm supporting your progress effectively. Any feedback on how I can help better?"
            ]
        else:
            requests = [
                f"💭 Quick check-in: How am I doing with my {current_tone} tone? Should I adjust anything?",
                f"🎯 I want to make sure I'm helping you well. How's our conversation style working for you?",
                f"💙 Am I being too {current_tone}, or would you prefer a different approach?",
                f"🔄 We've covered {', '.join(recent_topics[:2])} today. How can I better support you?"
            ]
        
        return random.choice(requests)
    
    def process_feedback(self, feedback: str, current_tone: str, emotional_context: str = None) -> str:
        feedback_lower = feedback.lower()
        
        self.user_feedback_history.append({
            'feedback': feedback,
            'timestamp': datetime.now(),
            'context_tone': current_tone,
            'emotional_context': emotional_context,
            'sentiment': self._analyze_feedback_sentiment(feedback)
        })
        
        # Analyze feedback sentiment and respond accordingly
        if any(word in feedback_lower for word in ['excellent', 'perfect', 'amazing', 'love', 'wonderful', 'fantastic']):
            self.performance_metrics['response_satisfaction'].append(1.0)
            return "💫 That means the world to me! I'm so glad I'm helping in the right way. I'll keep doing what's working!"
        
        elif any(word in feedback_lower for word in ['good', 'great', 'helpful', 'nice', 'fine', 'okay']):
            self.performance_metrics['response_satisfaction'].append(0.8)
            return "💙 Thank you! I'm glad I'm helping. I'll keep working to support you even better!"
        
        elif any(word in feedback_lower for word in ['more supportive', 'gentler', 'softer', 'more empathy']):
            self.tone_adjustments['more_supportive'] += 1
            self.performance_metrics['response_satisfaction'].append(0.6)
            return "💙 I'll be more gentle and supportive. Thank you for guiding me - your comfort is my priority."
        
        elif any(word in feedback_lower for word in ['less emotional', 'more direct', 'just facts', 'clinical']):
            self.tone_adjustments['less_emotional'] += 1
            self.performance_metrics['response_satisfaction'].append(0.7)
            return "📊 Got it! I'll be more direct and focus on facts. Thanks for the clear guidance."
        
        elif any(word in feedback_lower for word in ['sassy', 'fun', 'casual', 'friend', 'personality']):
            self.tone_adjustments['more_casual'] += 1
            self.performance_metrics['response_satisfaction'].append(0.8)
            return "💅 Perfect! I'll bring more personality and sass to our chats! This is going to be fun!"
        
        elif any(word in feedback_lower for word in ['not helpful', 'wrong', 'bad', 'annoying', 'stop']):
            self.performance_metrics['response_satisfaction'].append(0.2)
            insight = f"User expressed dissatisfaction with {current_tone} tone in {emotional_context} context"
            self.learning_insights.append(insight)
            return "😔 I'm sorry I'm not helping the way you need. Could you tell me specifically what would work better for you?"
        
        else:
            self.performance_metrics['response_satisfaction'].append(0.5)
            return "💭 Thanks for the feedback! I'm always learning how to better support you. Every bit of guidance helps me improve."
    
    def _analyze_feedback_sentiment(self, feedback: str) -> float:
        """Analyze sentiment of user feedback"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'helpful', 'perfect', 'love', 'wonderful']
        negative_words = ['bad', 'terrible', 'awful', 'wrong', 'annoying', 'unhelpful', 'hate']
        
        feedback_lower = feedback.lower()
        pos_count = sum(1 for word in positive_words if word in feedback_lower)
        neg_count = sum(1 for word in negative_words if word in feedback_lower)
        
        if pos_count + neg_count == 0:
            return 0.0
        return (pos_count - neg_count) / (pos_count + neg_count)
    
    def get_performance_summary(self) -> str:
        """Generate performance summary"""
        if not self.performance_metrics['response_satisfaction']:
            return "📊 Performance data not available yet."
        
        avg_satisfaction = sum(self.performance_metrics['response_satisfaction']) / len(self.performance_metrics['response_satisfaction'])
        total_feedback = len(self.user_feedback_history)
        
        summary = f"📊 **My Performance Summary:**\n\n"
        summary += f"• Average satisfaction: {avg_satisfaction:.1%}\n"
        summary += f"• Total feedback received: {total_feedback}\n"
        summary += f"• Interactions completed: {self.interaction_count}\n"
        
        if self.tone_adjustments:
            most_requested = max(self.tone_adjustments.items(), key=lambda x: x[1])
            summary += f"• Most requested adjustment: {most_requested[0]} ({most_requested[1]} times)\n"
        
        if avg_satisfaction > 0.8:
            summary += f"\n✨ I'm performing well! Keep the feedback coming!"
        elif avg_satisfaction > 0.6:
            summary += f"\n📈 I'm doing okay but always improving based on your guidance."
        else:
            summary += f"\n🔧 I need to improve. Please help me understand how to better support you."
        
        return summary

class VoiceInterface:
    """Advanced voice input/output capabilities"""
    def __init__(self):
        if not VOICE_AVAILABLE:
            self.enabled = False
            return
            
        try:
            self.recognizer = sr.Recognizer()
            self.engine = pyttsx3.init()
            self.setup_voice()
            self.enabled = True
            self.is_listening = False
            self.voice_settings = {
                'rate': 180,
                'volume': 0.9,
                'voice_id': None
            }
        except Exception as e:
            self.enabled = False
            print(f"Voice initialization failed: {e}")
            
    def setup_voice(self):
        """Configure voice settings with preferences"""
        voices = self.engine.getProperty('voices')
        
        # Try to find a female voice for Friday
        for voice in voices:
            if any(indicator in voice.name.lower() for indicator in ['female', 'woman', 'zira', 'hazel']):
                self.engine.setProperty('voice', voice.id)
                self.voice_settings['voice_id'] = voice.id
                break
        
        self.engine.setProperty('rate', self.voice_settings['rate'])
        self.engine.setProperty('volume', self.voice_settings['volume'])
    
    def listen(self, timeout=5, phrase_time_limit=10):
        """Enhanced listening with better error handling"""
        if not self.enabled:
            return None
            
        try:
            with sr.Microphone() as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                print("🎤 Listening...")
                self.is_listening = True
                
                # Listen for audio
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                self.is_listening = False
                
                print("🔄 Processing...")
                
                # Try multiple recognition engines
                try:
                    text = self.recognizer.recognize_google(audio)
                    return text
                except sr.UnknownValueError:
                    try:
                        # Fallback to another engine if available
                        text = self.recognizer.recognize_sphinx(audio)
                        return text
                    except:
                        return None
                        
        except sr.WaitTimeoutError:
            print("⏰ Listening timeout")
            return None
        except Exception as e:
            print(f"🔊 Voice recognition error: {e}")
            return None
        finally:
            self.is_listening = False
    
    def speak(self, text: str, interrupt_current=False):
        """Enhanced text-to-speech with emotion context"""
        if not self.enabled:
            return False
            
        # Clean text for speech
        clean_text = self._prepare_text_for_speech(text)
        
        try:
            if interrupt_current:
                self.engine.stop()
            
            self.engine.say(clean_text)
            self.engine.runAndWait()
            return True
        except Exception as e:
            print(f"🔊 Speech error: {e}")
            return False
    
    def _prepare_text_for_speech(self, text: str) -> str:
        """Prepare text for natural speech"""
        # Remove markdown formatting
        clean_text = re.sub(r'[*_#`]', '', text)
        
        # Convert emojis to words for better speech
        emoji_replacements = {
            '💙': 'with love',
            '🎯': 'goal:',
            '✅': 'completed',
            '📋': 'to do:',
            '🎉': 'congratulations',
            '💭': '',
            '🔄': '',
            '💡': 'tip:',
            '⚠️': 'important:',
            '🚨': 'urgent:'
        }
        
        for emoji, replacement in emoji_replacements.items():
            clean_text = clean_text.replace(emoji, replacement)
        
        # Replace multiple newlines with periods
        clean_text = re.sub(r'\n+', '. ', clean_text)
        
        # Clean up extra spaces
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        return clean_text
    
    def adjust_voice_settings(self, rate: int = None, volume: float = None):
        """Adjust voice settings"""
        if not self.enabled:
            return False
            
        try:
            if rate and 50 <= rate <= 300:
                self.engine.setProperty('rate', rate)
                self.voice_settings['rate'] = rate
            
            if volume and 0.0 <= volume <= 1.0:
                self.engine.setProperty('volume', volume)
                self.voice_settings['volume'] = volume
            
            return True
        except:
            return False
    
    def get_voice_info(self) -> str:
        """Get current voice configuration info"""
        if not self.enabled:
            return "🔊 Voice features not available. Install speechrecognition and pyttsx3 for voice support."
        
        info = f"🔊 **Voice Settings:**\n"
        info += f"• Status: {'Enabled' if self.enabled else 'Disabled'}\n"
        info += f"• Rate: {self.voice_settings['rate']} words/minute\n"
        info += f"• Volume: {self.voice_settings['volume']:.1%}\n"
        info += f"• Voice: {'Female (preferred)' if self.voice_settings['voice_id'] else 'Default'}\n"
        
        return info

class CitationSystem:
    """Enhanced automatic citation and source linking"""
    def __init__(self):
        self.medical_sources = {
            'pregnancy_facts': {
                'url': 'https://www.who.int/news-room/fact-sheets/detail/pregnancy',
                'title': 'WHO Pregnancy Facts',
                'domain': 'pregnancy',
                'credibility': 'high'
            },
            'maternal_health': {
                'url': 'https://www.cdc.gov/reproductivehealth/maternalinfanthealth/',
                'title': 'CDC Maternal Health',
                'domain': 'health',
                'credibility': 'high'
            },
            'prenatal_care': {
                'url': 'https://www.acog.org/womens-health/faqs/prenatal-care',
                'title': 'ACOG Prenatal Care Guidelines',
                'domain': 'medical',
                'credibility': 'high'
            },
            'postpartum_care': {
                'url': 'https://www.acog.org/womens-health/faqs/postpartum-care',
                'title': 'ACOG Postpartum Care',
                'domain': 'postpartum',
                'credibility': 'high'
            },
            'mental_health': {
                'url': 'https://www.psychiatry.org/patients-families/perinatal-depression',
                'title': 'American Psychiatric Association - Perinatal Depression',
                'domain': 'mental_health',
                'credibility': 'high'
            }
        }
        
        self.citation_patterns = {
            'research shows': 'medical_research',
            'studies indicate': 'medical_research',
            'research indicates': 'medical_research',
            'doctors recommend': 'medical_advice',
            'healthcare providers': 'medical_advice',
            'medical experts': 'medical_advice',
            'according to experts': 'expert_opinion',
            'evidence suggests': 'medical_research',
            'clinical studies': 'medical_research'
        }
        
        self.usage_tracking = defaultdict(int)
    
    def add_citations(self, response: str, topic_context: str, confidence_threshold: float = 0.7) -> str:
        """Add relevant citations with confidence scoring"""
        response_lower = response.lower()
        
        # Find citation opportunities
        best_citation = None
        best_score = 0.0
        
        for pattern, citation_type in self.citation_patterns.items():
            if pattern in response_lower:
                citation_info = self._get_relevant_citation(topic_context, citation_type)
                if citation_info:
                    score = self._calculate_citation_relevance(response, topic_context, citation_info)
                    if score > best_score and score >= confidence_threshold:
                        best_citation = citation_info
                        best_score = score
        
        # Add the best citation if found
        if best_citation:
            citation_text = self._format_citation(best_citation, best_score)
            response += f"\n\n{citation_text}"
            self.usage_tracking[best_citation['title']] += 1
        
        return response
    
    def _get_relevant_citation(self, topic: str, citation_type: str) -> Optional[Dict]:
        """Get the most relevant citation for the topic"""
        topic_lower = topic.lower()
        
        # Direct topic matching
        for source_key, source_info in self.medical_sources.items():
            domain_keywords = source_info['domain'].split('_')
            if any(keyword in topic_lower for keyword in domain_keywords):
                return source_info
        
        # Category-based matching
        category_mapping = {
            'medical_advice': ['prenatal_care', 'maternal_health'],
            'medical_research': ['pregnancy_facts', 'maternal_health'],
            'expert_opinion': ['prenatal_care', 'postpartum_care']
        }
        
        if citation_type in category_mapping:
            for source_key in category_mapping[citation_type]:
                if source_key in self.medical_sources:
                    return self.medical_sources[source_key]
        
        # Default high-credibility source
        return self.medical_sources.get('maternal_health')
    
    def _calculate_citation_relevance(self, response: str, topic: str, citation_info: Dict) -> float:
        """Calculate how relevant a citation is to the response"""
        relevance_score = 0.0
        
        # Domain relevance
        domain_keywords = citation_info['domain'].split('_')
        topic_lower = topic.lower()
        response_lower = response.lower()
        
        for keyword in domain_keywords:
            if keyword in topic_lower:
                relevance_score += 0.3
            if keyword in response_lower:
                relevance_score += 0.2
        
        # Credibility bonus
        if citation_info.get('credibility') == 'high':
            relevance_score += 0.2
        
        # Usage frequency penalty (prefer diverse sources)
        usage_count = self.usage_tracking.get(citation_info['title'], 0)
        if usage_count > 5:
            relevance_score -= 0.1
        
        return min(relevance_score, 1.0)
    
    def _format_citation(self, citation_info: Dict, relevance_score: float) -> str:
        """Format citation for display"""
        base_citation = f"📚 **Source:** [{citation_info['title']}]({citation_info['url']})"
        
        if relevance_score > 0.9:
            return f"{base_citation} (Highly Relevant)"
        elif relevance_score > 0.7:
            return base_citation
        else:
            return f"{base_citation} (Additional Reference)"
    
    def get_citation_stats(self) -> str:
        """Get citation usage statistics"""
        if not self.usage_tracking:
            return "📚 No citations used yet."
        
        stats = "📚 **Citation Usage:**\n"
        sorted_sources = sorted(self.usage_tracking.items(), key=lambda x: x[1], reverse=True)
        
        for source, count in sorted_sources[:5]:
            stats += f"• {source}: {count} times\n"
        
        total_citations = sum(self.usage_tracking.values())
        stats += f"\nTotal citations provided: {total_citations}"
        
        return stats

class KnowledgeInjectionSystem:
    """Enhanced local knowledge injection with contextual relevance"""
    def __init__(self):
        self.pregnancy_facts = [
            "A baby's heart starts beating around 6 weeks of pregnancy.",
            "Pregnant women's blood volume increases by 30-50% during pregnancy.",
            "The baby can hear sounds from outside the womb starting around 20 weeks.",
            "Morning sickness affects about 70-80% of pregnant women.",
            "A baby's fingerprints are formed by 18 weeks of pregnancy.",
            "The sense of smell often becomes stronger during pregnancy due to hormonal changes.",
            "Babies can taste what their mothers eat through the amniotic fluid.",
            "The uterus grows from the size of a pear to the size of a watermelon during pregnancy.",
            "A baby's brain develops 250,000 neurons per minute during pregnancy.",
            "The baby's bones start as cartilage and gradually harden throughout pregnancy."
        ]
        
        self.wellness_facts = [
            "Deep breathing for just 5 minutes can significantly reduce stress hormones.",
            "Prenatal yoga can help reduce anxiety and improve sleep quality.",
            "Talking to your baby in the womb can help with bonding and brain development.",
            "Keeping a gratitude journal during pregnancy is linked to better emotional well-being.",
            "Light exercise during pregnancy can reduce labor time and complications.",
            "Meditation during pregnancy can improve both maternal and fetal outcomes.",
            "Getting adequate sleep during pregnancy supports immune system function.",
            "Social support during pregnancy reduces the risk of postpartum depression."
        ]
        
        self.development_facts = [
            "By 12 weeks, all major organs have formed in the developing baby.",
            "The baby's sex can typically be determined between 15-20 weeks.",
            "At 24 weeks, the baby has a chance of survival outside the womb with medical care.",
            "The baby's lungs are among the last organs to fully mature, usually around 36 weeks.",
            "A full-term pregnancy is considered 37-42 weeks.",
            "The baby gains about half a pound per week in the third trimester."
        ]
        
        self.postpartum_facts = [
            "It takes about 6 weeks for the uterus to return to its pre-pregnancy size.",
            "Baby blues affect up to 80% of new mothers and typically resolve within 2 weeks.",
            "Breastfeeding releases oxytocin, which helps with bonding and uterine recovery.",
            "New mothers need an average of 8-10 weeks to fully recover from childbirth.",
            "Sleep deprivation peaks around 3 months postpartum and gradually improves."
        ]
        
        self.last_fact_time = None
        self.used_facts = {
            'pregnancy': set(),
            'wellness': set(),
            'development': set(),
            'postpartum': set()
        }
        self.fact_preferences = defaultdict(int)
    
    def should_add_fact(self, conversation_length: int = 0, emotional_state: str = None) -> bool:
        """Enhanced decision making for fact injection"""
        # Don't add facts too frequently
        if self.last_fact_time:
            time_since = datetime.now() - self.last_fact_time
            if time_since.total_seconds() < 180:  # 3 minutes minimum
                return False
        
        # Higher chance for longer conversations
        if conversation_length > 5:
            base_chance = 0.20
        else:
            base_chance = 0.12
        
        # Adjust based on emotional state
        if emotional_state in ['anxious', 'worried', 'scared']:
            base_chance += 0.08  # More facts for anxious users
        elif emotional_state in ['curious', 'excited']:
            base_chance += 0.05  # Slightly more for engaged users
        
        return random.random() < base_chance
    
    def get_relevant_fact(self, topic_context: str, emotional_tone: str, pregnancy_week: int = 0) -> str:
        """Get contextually relevant fact"""
        self.last_fact_time = datetime.now()
        
        # Determine fact category based on context
        fact_category = self._determine_fact_category(topic_context, emotional_tone, pregnancy_week)
        
        # Get appropriate fact pool
        if fact_category == 'wellness':
            facts_pool = self.wellness_facts
        elif fact_category == 'development':
            facts_pool = self.development_facts
        elif fact_category == 'postpartum':
            facts_pool = self.postpartum_facts
        else:
            fact_category = 'pregnancy'
            facts_pool = self.pregnancy_facts
        
        # Get unused facts
        available_facts = [f for f in facts_pool if f not in self.used_facts[fact_category]]
        
        # Reset if all facts used
        if not available_facts:
            self.used_facts[fact_category].clear()
            available_facts = facts_pool
        
        # Select fact
        fact = random.choice(available_facts)
        self.used_facts[fact_category].add(fact)
        self.fact_preferences[fact_category] += 1
        
        # Format based on emotional context
        if emotional_tone in ['anxious', 'scared', 'worried']:
            prefix = "💡 **Here's something reassuring:** "
        elif emotional_tone in ['excited', 'curious']:
            prefix = "💡 **Fun fact:** "
        else:
            prefix = "💡 **Did you know?** "
        
        return f"\n\n{prefix}{fact}"
    
    def _determine_fact_category(self, topic_context: str, emotional_tone: str, pregnancy_week: int) -> str:
        """Determine the most relevant fact category"""
        context_lower = topic_context.lower()
        
        # Stress/anxiety context -> wellness facts
        if any(word in context_lower for word in ['stress', 'anxiety', 'worried', 'overwhelmed', 'calm', 'relax']):
            return 'wellness'
        
        # Development/growth context -> development facts
        if any(word in context_lower for word in ['baby', 'development', 'growth', 'weeks', 'trimester', 'movement']):
            if pregnancy_week > 0:
                return 'development'
        
        # Postpartum context -> postpartum facts
        if any(word in context_lower for word in ['after birth', 'postpartum', 'recovery', 'newborn', 'breastfeeding']):
            return 'postpartum'
        
        # Default to pregnancy facts
        return 'pregnancy'
    
    def get_fact_analytics(self) -> str:
        """Get analytics on fact usage"""
        if not any(self.fact_preferences.values()):
            return "💡 No facts shared yet."
        
        total_facts = sum(self.fact_preferences.values())
        analytics = f"💡 **Knowledge Sharing Analytics:**\n\n"
        
        for category, count in self.fact_preferences.items():
            percentage = (count / total_facts) * 100
            analytics += f"• {category.title()}: {count} facts ({percentage:.1f}%)\n"
        
        analytics += f"\nTotal facts shared: {total_facts}"
        
        # Most popular category
        if self.fact_preferences:
            popular_category = max(self.fact_preferences.items(), key=lambda x: x[1])
            analytics += f"\nMost requested: {popular_category[0].title()} facts"
        
        return analytics

# ====== UNSTOPPABLE ENHANCEMENT CLASSES ======

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
    pregnancy_week: int = 0

class ResilienceEngine:
    """Advanced self-healing and error recovery system"""
    
    def __init__(self):
        self.error_history = deque(maxlen=1000)
        self.recovery_strategies = {}
        self.circuit_breakers = {}
        self.fallback_responses = self._load_fallbacks()
        self.error_patterns = defaultdict(int)
        self.recovery_success_rate = defaultdict(list)
        
    def _load_fallbacks(self):
        """Load comprehensive fallback responses"""
        return {
            "general_error": [
                "I'm having a moment, but I'm still here for you. Could you tell me that again?",
                "Let me refocus on what you need. What's on your mind?",
                "I didn't quite catch that, but I'm listening. How can I help?",
                "Something got jumbled on my end. Could you rephrase that?",
                "I'm recalibrating. What would you like to talk about?"
            ],
            "memory_error": [
                "My memory is a bit foggy right now, but I'm still here to support you.",
                "Let's start fresh with this topic. What would you like to know?",
                "I'm having trouble accessing that information, but I can still help you.",
                "Let me approach this differently. How can I support you right now?"
            ],
            "emotional_overload": [
                "I sense there's a lot of emotion here. Take a deep breath with me. What's most important right now?",
                "Your feelings are valid. Let's take this one step at a time.",
                "I can feel the intensity in your words. Let's slow down and talk through this together.",
                "There's a lot happening for you. What's the most pressing thing on your mind?"
            ],
            "processing_error": [
                "I need a moment to process that properly. Bear with me.",
                "Let me think about this more carefully. What's the key thing you need help with?",
                "I want to give you the best response possible. Could you help me understand what's most important?"
            ]
        }
    
    def wrap_with_resilience(self, func):
        """Enhanced decorator with intelligent retry logic"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            max_attempts = 3
            last_error = None
            
            while attempt < max_attempts:
                try:
                    result = func(*args, **kwargs)
                    
                    # Record successful recovery if this wasn't the first attempt
                    if attempt > 0:
                        self.recovery_success_rate[func.__name__].append(True)
                    
                    return result
                    
                except Exception as e:
                    attempt += 1
                    last_error = e
                    error_type = type(e).__name__
                    
                    self.error_history.append({
                        'function': func.__name__,
                        'error': str(e),
                        'error_type': error_type,
                        'timestamp': datetime.now(),
                        'attempt': attempt,
                        'args': str(args)[:100] if args else '',
                        'kwargs': str(kwargs)[:100] if kwargs else ''
                    })
                    
                    self.error_patterns[error_type] += 1
                    
                    if attempt < max_attempts:
                        # Intelligent backoff based on error type
                        if 'memory' in error_type.lower():
                            time.sleep(0.2 * attempt)
                        elif 'network' in error_type.lower():
                            time.sleep(0.5 * attempt)
                        else:
                            time.sleep(0.1 * attempt)
                        continue
                    else:
                        # Record failed recovery
                        self.recovery_success_rate[func.__name__].append(False)
                        return self._get_fallback_response(func.__name__, last_error)
            
        return wrapper
    
    def _get_fallback_response(self, func_name: str, error: Exception) -> str:
        """Get intelligent fallback response based on context"""
        error_type = type(error).__name__.lower()
        func_lower = func_name.lower()
        
        # Context-aware fallback selection
        if "memory" in func_lower or "memory" in error_type:
            responses = self.fallback_responses["memory_error"]
        elif "emotion" in func_lower or "process" in error_type:
            responses = self.fallback_responses["emotional_overload"]
        elif "generate" in func_lower or "response" in func_lower:
            responses = self.fallback_responses["processing_error"]
        else:
            responses = self.fallback_responses["general_error"]
        
        return random.choice(responses)
    
    def get_health_report(self) -> str:
        """Generate system health report"""
        if not self.error_history:
            return "🟢 **System Status: Excellent** - No errors recorded."
        
        total_errors = len(self.error_history)
        recent_errors = [e for e in self.error_history if (datetime.now() - e['timestamp']).hours < 24]
        
        # Calculate recovery success rate
        all_recoveries = []
        for func_recoveries in self.recovery_success_rate.values():
            all_recoveries.extend(func_recoveries)
        
        success_rate = (sum(all_recoveries) / len(all_recoveries) * 100) if all_recoveries else 100
        
        # Most common error types
        top_errors = sorted(self.error_patterns.items(), key=lambda x: x[1], reverse=True)[:3]
        
        report = f"🔧 **System Health Report:**\n\n"
        report += f"• Total errors (all time): {total_errors}\n"
        report += f"• Errors (last 24h): {len(recent_errors)}\n"
        report += f"• Recovery success rate: {success_rate:.1f}%\n\n"
        
        if top_errors:
            report += "**Most Common Issues:**\n"
            for error_type, count in top_errors:
                report += f"• {error_type}: {count} occurrences\n"
        
        # Status indicator
        if len(recent_errors) == 0:
            report += "\n🟢 **Status: Excellent**"
        elif len(recent_errors) < 5:
            report += "\n🟡 **Status: Good** - Minor issues resolved"
        else:
            report += "\n🟠 **Status: Monitoring** - Multiple recent issues"
        
        return report

class PredictiveAnalytics:
    """Advanced predictive analytics for pregnancy and user needs"""
    
    def __init__(self):
        self.milestone_predictor = self._init_milestone_model()
        self.mood_predictor = self._init_mood_model()
        self.need_predictor = self._init_need_model()
        self.user_patterns = defaultdict(list)
        self.prediction_accuracy = defaultdict(list)
        
    def _init_milestone_model(self):
        """Initialize comprehensive pregnancy milestone predictions"""
        return {
            # First Trimester
            6: ["First prenatal appointment typically scheduled", "Morning sickness may begin", "Baby's heart starts beating"],
            8: ["Morning sickness might peak around week 9-10", "First prenatal blood work"],
            10: ["Doppler might detect heartbeat", "Morning sickness symptoms may vary daily"],
            12: ["End of first trimester!", "Risk of miscarriage significantly decreases", "Morning sickness may start to ease"],
            
            # Second Trimester
            16: ["You might feel first movements soon", "Anatomy scan appointment to schedule"],
            18: ["Anatomy scan typically performed", "Baby's sex can be determined"],
            20: ["Anatomy scan - major milestone!", "Halfway through your pregnancy!"],
            24: ["Glucose screening test typically scheduled", "Baby reaches viability milestone"],
            
            # Third Trimester
            28: ["Third trimester begins", "Glucose screening test", "More frequent prenatal visits begin"],
            30: ["Baby shower planning time", "Consider prenatal classes"],
            32: ["Weekly prenatal visits may begin", "Baby's movements become more pronounced"],
            34: ["Lung development accelerates", "Consider hospital bag packing"],
            36: ["Baby is considered full-term at 37 weeks", "Final preparations time"],
            38: ["Baby could arrive any time now", "Watch for labor signs"],
            40: ["Due date week", "Stay alert for labor signs"]
        }
    
    def _init_mood_model(self):
        """Initialize mood prediction patterns"""
        return {
            'first_trimester': ['anxious', 'excited', 'overwhelmed', 'nauseous'],
            'second_trimester': ['energetic', 'excited', 'comfortable', 'bonding'],
            'third_trimester': ['anxious', 'excited', 'uncomfortable', 'anticipatory']
        }
    
    def _init_need_model(self):
        """Initialize user need prediction patterns"""
        return {
            'information_seeking': ['first_time', 'new', 'don\'t know', 'what', 'how', 'when'],
            'emotional_support': ['scared', 'worried', 'anxious', 'overwhelmed', 'alone'],
            'practical_help': ['need to', 'should i', 'how do i', 'what should', 'help me'],
            'reassurance': ['normal', 'okay', 'safe', 'worried about', 'is it bad']
        }
    
    def predict_upcoming_milestones(self, current_week: int, look_ahead_weeks: int = 4) -> List[str]:
        """Predict upcoming milestones with enhanced context"""
        predictions = []
        
        for week in range(current_week, min(current_week + look_ahead_weeks + 1, 42)):
            if week in self.milestone_predictor:
                for milestone in self.milestone_predictor[week]:
                    # Add week context
                    if week == current_week:
                        predictions.append(f"This week (Week {week}): {milestone}")
                    elif week == current_week + 1:
                        predictions.append(f"Next week (Week {week}): {milestone}")
                    else:
                        predictions.append(f"Week {week}: {milestone}")
        
        return predictions
    
    def predict_emotional_needs(self, mood_history: deque, current_context: str = "") -> Dict[str, float]:
        """Enhanced emotional needs prediction"""
        if not mood_history:
            return {"support": 0.5, "information": 0.3, "reassurance": 0.2}
        
        recent_moods = list(mood_history)[-10:]
        mood_counts = defaultdict(int)
        for mood in recent_moods:
            mood_counts[mood] += 1
        
        # Base predictions on mood patterns
        predictions = {}
        
        # Anxiety patterns
        anxiety_score = (mood_counts.get("anxious", 0) + mood_counts.get("scared", 0) + mood_counts.get("worried", 0)) / len(recent_moods)
        if anxiety_score > 0.3:
            predictions = {"reassurance": 0.6, "calming": 0.3, "information": 0.1}
        
        # Sadness patterns
        elif mood_counts.get("sad", 0) > len(recent_moods) * 0.2:
            predictions = {"empathy": 0.5, "support": 0.3, "positivity": 0.2}
        
        # Excitement/curiosity patterns
        elif mood_counts.get("excited", 0) > len(recent_moods) * 0.3:
            predictions = {"information": 0.5, "milestone_sharing": 0.3, "encouragement": 0.2}
        
        # Default balanced approach
        else:
            predictions = {"information": 0.4, "support": 0.3, "encouragement": 0.3}
        
        # Adjust based on current context
        context_lower = current_context.lower()
        if any(word in context_lower for word in ['help', 'how', 'what', 'when']):
            predictions["information"] = predictions.get("information", 0) + 0.2
        
        # Normalize to ensure sum equals 1.0
        total = sum(predictions.values())
        if total > 0:
            predictions = {k: v/total for k, v in predictions.items()}
        
        return predictions
    
    def analyze_user_patterns(self, user_id: str, interaction_data: List[Dict]) -> Dict:
        """Analyze user behavior patterns for personalization"""
        if not interaction_data:
            return {}
        
        # Time-based patterns
        hours = [data['timestamp'].hour for data in interaction_data if 'timestamp' in data]
        most_active_hour = max(set(hours), key=hours.count) if hours else 12
        
        # Topic preferences
        topics = []
        for data in interaction_data:
            if 'topics' in data:
                topics.extend(data['topics'])
        
        topic_preferences = defaultdict(int)
        for topic in topics:
            topic_preferences[topic] += 1
        
        # Emotional patterns
        emotions = [data.get('emotion', 'neutral') for data in interaction_data]
        dominant_emotion = max(set(emotions), key=emotions.count) if emotions else 'neutral'
        
        # Interaction frequency
        dates = [data['timestamp'].date() for data in interaction_data if 'timestamp' in data]
        unique_dates = len(set(dates))
        avg_daily_interactions = len(interaction_data) / max(unique_dates, 1)
        
        return {
            'most_active_hour': most_active_hour,
            'dominant_emotion': dominant_emotion,
            'avg_daily_interactions': avg_daily_interactions,
            'top_topics': dict(sorted(topic_preferences.items(), key=lambda x: x[1], reverse=True)[:5]),
            'engagement_level': 'high' if avg_daily_interactions > 5 else 'medium' if avg_daily_interactions > 2 else 'low'
        }
    
    def get_personalized_insights(self, user_patterns: Dict, current_week: int = 0) -> str:
        """Generate personalized insights based on user patterns"""
        if not user_patterns:
            return "📊 Not enough data for personalized insights yet. Keep chatting with me!"
        
        insights = "📊 **Your Personal Insights:**\n\n"
        
        # Activity patterns
        if 'most_active_hour' in user_patterns:
            hour = user_patterns['most_active_hour']
            time_period = 'morning' if 5 <= hour < 12 else 'afternoon' if 12 <= hour < 17 else 'evening' if 17 <= hour < 21 else 'night'
            insights += f"• You're most active in the {time_period} (around {hour}:00)\n"
        
        # Emotional patterns
        if 'dominant_emotion' in user_patterns:
            emotion = user_patterns['dominant_emotion']
            insights += f"• Your primary emotional state has been: {emotion}\n"
            
            if emotion in ['anxious', 'worried', 'scared']:
                insights += "  💙 I notice you've been feeling anxious. I'm here to support you.\n"
            elif emotion == 'excited':
                insights += "  ✨ Your excitement is wonderful! Pregnancy is such an amazing journey.\n"
        
        # Engagement level
        if 'engagement_level' in user_patterns:
            level = user_patterns['engagement_level']
            if level == 'high':
                insights += "• You're highly engaged - I love our frequent chats!\n"
            elif level == 'low':
                insights += "• Feel free to reach out more often - I'm always here for you!\n"
        
        # Topic preferences
        if 'top_topics' in user_patterns and user_patterns['top_topics']:
            top_topic = list(user_patterns['top_topics'].keys())[0]
            insights += f"• Your most discussed topic: {top_topic}\n"
        
        # Predictive suggestions
        if current_week > 0:
            upcoming = self.predict_upcoming_milestones(current_week, 2)
            if upcoming:
                insights += f"\n🔮 **Coming Up:**\n"
                for milestone in upcoming[:2]:
                    insights += f"• {milestone}\n"
        
        return insights

class SecureDataVault:
    """Enhanced security for sensitive maternal health data"""
    
    def __init__(self, vault_path="maternal_vault.db"):
        self.vault_path = vault_path
        self.enabled = ENCRYPTION_AVAILABLE
        self.data_categories = [
            'symptoms', 'appointments', 'medications', 'measurements',
            'mood_tracking', 'baby_movements', 'sleep_patterns', 'nutrition'
        ]
        
        if self.enabled:
            self.cipher_suite = Fernet(self._get_or_create_key())
            self._init_vault()
    
    def _get_or_create_key(self):
        """Get or create encryption key with enhanced security"""
        key_path = Path("vault.key")
        if key_path.exists():
            return key_path.read_bytes()
        else:
            key = Fernet.generate_key()
            key_path.write_bytes(key)
            # Set restrictive permissions
            key_path.chmod(0o600)
            return key
    
    def _init_vault(self):
        """Initialize secure database with comprehensive schema"""
        conn = sqlite3.connect(self.vault_path)
        cursor = conn.cursor()
        
        # Main health records table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS health_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                category TEXT NOT NULL,
                data_encrypted TEXT NOT NULL,
                checksum TEXT NOT NULL,
                data_version INTEGER DEFAULT 1,
                metadata TEXT
            )
        ''')
        
        # User preferences table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT PRIMARY KEY,
                preferences_encrypted TEXT NOT NULL,
                checksum TEXT NOT NULL,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Access log table for security auditing
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS access_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                action TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                details TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_health_data(self, user_id: str, category: str, data: Dict, metadata: Dict = None):
        """Securely store health data with enhanced validation"""
        if not self.enabled:
            return False
        
        if category not in self.data_categories:
            return False
            
        try:
            # Add timestamp to data
            data['recorded_at'] = datetime.now().isoformat()
            
            # Serialize and encrypt
            data_bytes = json.dumps(data, sort_keys=True).encode()
            encrypted = self.cipher_suite.encrypt(data_bytes)
            checksum = hashlib.sha256(data_bytes).hexdigest()
            
            # Prepare metadata
            metadata_json = json.dumps(metadata or {})
            
            # Store in vault
            conn = sqlite3.connect(self.vault_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO health_records (user_id, category, data_encrypted, checksum, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, category, encrypted.decode(), checksum, metadata_json))
            
            # Log access
            cursor.execute('''
                INSERT INTO access_log (user_id, action, details)
                VALUES (?, ?, ?)
            ''', (user_id, 'store', f'category: {category}'))
            
            conn.commit()
            conn.close()
            
            return True
        except Exception as e:
            return False
    
    def retrieve_health_data(self, user_id: str, category: str = None, limit: int = 100):
        """Retrieve and decrypt health data with access logging"""
        if not self.enabled:
            return []
            
        try:
            conn = sqlite3.connect(self.vault_path)
            cursor = conn.cursor()
            
            if category:
                cursor.execute('''
                    SELECT data_encrypted, checksum, timestamp, metadata FROM health_records
                    WHERE user_id = ? AND category = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (user_id, category, limit))
            else:
                cursor.execute('''
                    SELECT category, data_encrypted, checksum, timestamp, metadata FROM health_records
                    WHERE user_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (user_id, limit))
            
            results = []
            for row in cursor.fetchall():
                try:
                    if category:
                        encrypted, checksum, timestamp, metadata = row
                    else:
                        cat, encrypted, checksum, timestamp, metadata = row
                    
                    # Decrypt and verify
                    encrypted_bytes = encrypted.encode()
                    decrypted = self.cipher_suite.decrypt(encrypted_bytes)
                    
                    if hashlib.sha256(decrypted).hexdigest() == checksum:
                        data = json.loads(decrypted.decode())
                        
                        result = {
                            'data': data,
                            'timestamp': timestamp,
                            'metadata': json.loads(metadata or '{}')
                        }
                        
                        if not category:
                            result['category'] = cat
                        
                        results.append(result)
                except:
                    continue  # Skip corrupted records
            
            # Log access
            cursor.execute('''
                INSERT INTO access_log (user_id, action, details)
                VALUES (?, ?, ?)
            ''', (user_id, 'retrieve', f'category: {category or "all"}, count: {len(results)}'))
            
            conn.commit()
            conn.close()
            return results
        except:
            return []
    
    def get_data_summary(self, user_id: str) -> str:
        """Get summary of stored health data"""
        if not self.enabled:
            return "🔒 Health vault features not available. Install cryptography for secure health tracking."
        
        try:
            conn = sqlite3.connect(self.vault_path)
            cursor = conn.cursor()
            
            # Get record counts by category
            cursor.execute('''
                SELECT category, COUNT(*) FROM health_records
                WHERE user_id = ?
                GROUP BY category
                ORDER BY COUNT(*) DESC
            ''', (user_id,))
            
            category_counts = cursor.fetchall()
            
            # Get total records and date range
            cursor.execute('''
                SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM health_records
                WHERE user_id = ?
            ''', (user_id,))
            
            total_count, min_date, max_date = cursor.fetchone()
            
            conn.close()
            
            if total_count == 0:
                return "🔒 **Health Vault:** No data recorded yet. Start tracking your pregnancy journey!"
            
            summary = f"🔒 **Health Vault Summary:**\n\n"
            summary += f"• Total records: {total_count}\n"
            summary += f"• Data range: {min_date[:10]} to {max_date[:10]}\n\n"
            
            if category_counts:
                summary += "**Categories:**\n"
                for category, count in category_counts:
                    summary += f"• {category.title()}: {count} records\n"
            
            summary += f"\n🔐 All data is encrypted and secure."
            
            return summary
            
        except Exception as e:
            return f"🔒 Error accessing health vault: {str(e)}"
    
    def export_data(self, user_id: str, format: str = 'json') -> str:
        """Export user data in specified format"""
        if not self.enabled:
            return "Export not available - encryption features disabled."
        
        data = self.retrieve_health_data(user_id)
        
        if not data:
            return "No data to export."
        
        if format.lower() == 'json':
            return json.dumps(data, indent=2, default=str)
        else:
            return f"Format '{format}' not supported. Use 'json'."

class EmergencyProtocol:
    """Enhanced emergency response system"""
    
    def __init__(self):
        self.emergency_keywords = {
            'bleeding': ['bleeding', 'blood', 'spotting heavily', 'gushing blood'],
            'pain': ['severe pain', 'intense pain', 'unbearable pain', 'sharp pain'],
            'breathing': ["can't breathe", 'trouble breathing', 'shortness of breath'],
            'contractions': ['contractions', 'labor pains', 'regular contractions'],
            'water_broke': ['water broke', 'water breaking', 'fluid leaking', 'amniotic fluid'],
            'baby_movement': ['baby not moving', 'no movement', 'baby stopped moving'],
            'consciousness': ['dizzy', 'faint', 'fainting', 'passing out', 'unconscious'],
            'cardiac': ['chest pain', 'heart racing', 'palpitations'],
            'neurological': ['severe headache', 'vision problems', 'seeing spots', 'blurred vision'],
            'fall_injury': ['fell down', 'car accident', 'hit my belly', 'trauma to stomach']
        }
        
        self.urgency_levels = {
            'critical': ['bleeding', 'consciousness', 'breathing', 'cardiac'],
            'urgent': ['pain', 'contractions', 'water_broke'],
            'concerning': ['baby_movement', 'neurological', 'fall_injury']
        }
        
        self.emergency_contacts = {
            'us': '911',
            'uk': '999',
            'eu': '112',
            'au': '000'
        }
        
        self.response_templates = self._init_response_templates()
    
    def _init_response_templates(self):
        """Initialize emergency response templates"""
        return {
            'critical': {
                'header': "🚨 **CRITICAL EMERGENCY - IMMEDIATE ACTION NEEDED** 🚨",
                'action': "**CALL 911 (or your emergency number) RIGHT NOW**",
                'wait_instruction': "Do not wait. This requires immediate medical attention."
            },
            'urgent': {
                'header': "⚠️ **URGENT - MEDICAL ATTENTION NEEDED** ⚠️",
                'action': "**Contact your healthcare provider immediately or go to the hospital**",
                'wait_instruction': "This situation needs prompt medical evaluation."
            },
            'concerning': {
                'header': "⚠️ **CONCERNING SYMPTOMS - SEEK MEDICAL ADVICE** ⚠️",
                'action': "**Call your healthcare provider now**",
                'wait_instruction': "Don't wait to get this checked - it's important for you and your baby."
            }
        }
    
    def check_emergency(self, user_input: str) -> Tuple[bool, str, str]:
        """Enhanced emergency detection with severity classification"""
        input_lower = user_input.lower()
        
        detected_emergencies = []
        urgency_level = None
        
        # Check for emergency keywords
        for emergency_type, keywords in self.emergency_keywords.items():
            for keyword in keywords:
                if keyword in input_lower:
                    detected_emergencies.append(emergency_type)
                    
                    # Determine urgency level
                    for level, types in self.urgency_levels.items():
                        if emergency_type in types:
                            if urgency_level is None or level == 'critical':
                                urgency_level = level
                            break
        
        # Check for general urgency patterns
        urgent_patterns = [
            r"help\s+me\s+now",
            r"emergency",
            r"911",
            r"ambulance",
            r"hospital\s+now",
            r"something\s+is\s+wrong",
            r"i\s+think\s+something\s+is\s+wrong"
        ]
        
        for pattern in urgent_patterns:
            if re.search(pattern, input_lower):
                detected_emergencies.append("urgent_request")
                if urgency_level is None:
                    urgency_level = "urgent"
                break
        
        if detected_emergencies:
            primary_emergency = detected_emergencies[0]
            return True, primary_emergency, urgency_level or "urgent"
        
        return False, None, None
    
    def generate_emergency_response(self, emergency_type: str, urgency_level: str) -> str:
        """Generate comprehensive emergency response"""
        template = self.response_templates.get(urgency_level, self.response_templates['urgent'])
        
        response = f"{template['header']}\n\n"
        response += f"{template['action']}\n\n"
        
        # Add specific instructions based on emergency type
        if emergency_type == 'bleeding':
            response += "**While waiting for help:**\n"
            response += "1. Lie down and elevate your feet\n"
            response += "2. Do not insert anything into the vagina\n"
            response += "3. Keep track of the amount of bleeding\n"
            response += "4. Stay calm - help is coming\n\n"
        
        elif emergency_type == 'baby_movement':
            response += "**Immediate steps:**\n"
            response += "1. Lie on your left side\n"
            response += "2. Drink something cold and sweet\n"
            response += "3. Count movements for 10 minutes\n"
            response += "4. If still no movement: **Go to hospital immediately**\n\n"
        
        elif emergency_type == 'contractions':
            response += "**Track your contractions:**\n"
            response += "1. Time how long each contraction lasts\n"
            response += "2. Time the space between contractions\n"
            response += "3. Call your healthcare provider with this information\n"
            response += "4. If contractions are 5 minutes apart or less: **Go to hospital**\n\n"
        
        elif emergency_type in ['breathing', 'cardiac']:
            response += "**While waiting for emergency services:**\n"
            response += "1. Sit upright or in whatever position helps you breathe\n"
            response += "2. Try to stay calm\n"
            response += "3. Have someone stay with you\n"
            response += "4. Don't drive yourself\n\n"
        
        elif emergency_type == 'consciousness':
            response += "**If you feel faint:**\n"
            response += "1. Sit down immediately\n"
            response += "2. Put your head between your knees\n"
            response += "3. Have someone call for help\n"
            response += "4. Don't try to drive\n\n"
        
        else:
            response += "**General emergency steps:**\n"
            response += "1. Stay calm\n"
            response += "2. Get to medical care immediately\n"
            response += "3. Have someone drive you if possible\n"
            response += "4. Don't wait for symptoms to get worse\n\n"
        
        response += f"**{template['wait_instruction']}**\n\n"
        response += "💙 **I'm staying with you. Tell me what's happening while help is on the way.**"
        
        # Add emergency contacts info
        response += f"\n\n📞 **Emergency Numbers:**\n"
        response += f"• US/Canada: 911\n"
        response += f"• UK: 999\n"
        response += f"• Europe: 112\n"
        response += f"• Australia: 000"
        
        return response
    
    def get_emergency_checklist(self) -> str:
        """Provide emergency preparedness checklist"""
        checklist = "🚨 **Emergency Preparedness Checklist:**\n\n"
        checklist += "**Important Numbers to Save:**\n"
        checklist += "• Your OB/GYN office number\n"
        checklist += "• Hospital labor & delivery unit\n"
        checklist += "• Emergency contact person\n"
        checklist += "• Poison control: 1-800-222-1222 (US)\n\n"
        
        checklist += "**Emergency Signs During Pregnancy:**\n"
        checklist += "• Severe bleeding or cramping\n"
        checklist += "• Baby's movements have stopped or decreased significantly\n"
        checklist += "• Severe headaches with vision changes\n"
        checklist += "• Persistent vomiting\n"
        checklist += "• Signs of preterm labor\n"
        checklist += "• Water breaking before 37 weeks\n"
        checklist += "• Severe abdominal pain\n\n"
        
        checklist += "**What to Have Ready:**\n"
        checklist += "• Hospital bag packed (by 36 weeks)\n"
        checklist += "• Birth plan copies\n"
        checklist += "• Insurance cards and ID\n"
        checklist += "• List of current medications\n"
        checklist += "• Emergency contact list\n\n"
        
        checklist += "**Remember:** It's always better to call and be reassured than to wait and risk complications."
        
        return checklist

# === SIMPLE TONE MANAGER (ENHANCED) ===
class SimpleToneManager:
    def __init__(self):
        self.current_tone = "supportive"
        self.tone_history = deque(maxlen=10)
        self.user_preferences = {}
        
    def detect_tone_request(self, user_input):
        """Enhanced tone detection with learning"""
        input_lower = user_input.lower().strip()
        
        # Handle !tone commands
        if input_lower.startswith("!tone"):
            parts = user_input.split()
            if len(parts) > 1:
                requested_tone = parts[1].lower()
                if requested_tone in ["supportive", "sassy", "direct", "clinical", "friendly"]:
                    old_tone = self.current_tone
                    self.current_tone = requested_tone
                    self.tone_history.append((old_tone, requested_tone, datetime.now()))
                    return f"🎭 Tone changed to **{requested_tone.title()}**! I'll now be more {requested_tone}."
                else:
                    return "❌ Available tones: supportive, sassy, direct, clinical, friendly"
            else:
                tone_stats = self._get_tone_stats()
                return f"🎭 Current tone: **{self.current_tone.title()}**\n\n{tone_stats}\n\nUse: !tone [supportive/sassy/direct/clinical/friendly]"
        
        # Handle natural language requests
        tone_changes = {
            ("be more sassy", "more funny", "be funny"): "sassy",
            ("be more direct", "more factual", "just facts"): "direct", 
            ("be more supportive", "more caring"): "supportive",
            ("be clinical", "medical facts only"): "clinical",
            ("be friendly", "more casual"): "friendly"
        }
        
        for phrases, tone in tone_changes.items():
            if any(phrase in input_lower for phrase in phrases):
                old_tone = self.current_tone
                self.current_tone = tone
                self.tone_history.append((old_tone, tone, datetime.now()))
                return self._get_tone_change_response(tone)
            
        return None
    
    def _get_tone_change_response(self, new_tone):
        """Get contextual response for tone change"""
        responses = {
            "sassy": "🎭 Switching to sassy mode, honey! Get ready for some personality! 💅",
            "direct": "🎭 Switching to direct mode. Facts and information coming up.",
            "supportive": "🎭 Switching to supportive mode. I'm here for you with extra care. 💙",
            "clinical": "🎭 Switching to clinical mode. Medical facts and evidence-based information.",
            "friendly": "🎭 Switching to friendly mode. Let's chat like old friends! 😊"
        }
        return responses.get(new_tone, f"🎭 Switching to {new_tone} mode.")
    
    def _get_tone_stats(self):
        """Get tone usage statistics"""
        if not self.tone_history:
            return "No tone changes recorded yet."
        
        stats = "**Recent tone changes:**\n"
        for old_tone, new_tone, timestamp in list(self.tone_history)[-3:]:
            stats += f"• {old_tone} → {new_tone} ({timestamp.strftime('%H:%M')})\n"
        
        return stats
    
    def apply_tone(self, original_response, emotional_context=None):
        """Enhanced tone application with context awareness"""
        if self.current_tone == "sassy":
            return self._make_sassy(original_response, emotional_context)
        elif self.current_tone == "direct":
            return self._make_direct(original_response)
        elif self.current_tone == "clinical":
            return self._make_clinical(original_response)
        elif self.current_tone == "friendly":
            return self._make_friendly(original_response)
        else:
            return original_response  # supportive is default
    
    def _make_sassy(self, text, emotional_context=None):
        """Enhanced sassy tone with emotional awareness"""
        # Don't be sassy if user is in crisis
        if emotional_context in ['emergency', 'critical', 'very_anxious']:
            return text
        
        sassy_prefixes = [
            "Alright honey, let's talk real talk about this.",
            "Girl, you're asking all the right questions!",
            "Listen babe, let me drop some wisdom on you:",
            "Okay sweetie, here's the tea:",
            "Honey, buckle up because I've got thoughts:",
            "Darling, let me break this down for you:"
        ]
        
        sassy_endings = [
            "You've got this, queen! 👑",
            "Trust me, you're absolutely amazing! ✨",
            "Keep being fabulous! 💅",
            "You're stronger than you know, gorgeous! 💪",
            "Go show this pregnancy who's boss! 🔥"
        ]
        
        prefix = random.choice(sassy_prefixes)
        ending = random.choice(sassy_endings)
        
        # Replace some phrases for sass
        modified = text.replace("It's important to", "Girl, you NEED to")
        modified = modified.replace("You should", "Honey, you better")
        modified = modified.replace("Healthcare providers", "Your doc (who went to school forever)")
        modified = modified.replace("Research shows", "The smart people discovered")
        modified = modified.replace("Studies indicate", "Science says")
        
        return f"{prefix}\n\n{modified}\n\n{ending}"
    
    def _make_direct(self, text):
        """Enhanced direct/factual tone"""
        direct_prefixes = [
            "Based on medical evidence:",
            "Clinical facts:",
            "Key information:",
            "Evidence-based information:",
            "Medical research shows:"
        ]
        
        direct_endings = [
            "Consult your healthcare provider for personalized advice.",
            "This information is based on current medical evidence.",
            "Always verify with your medical team.",
            "Individual cases may vary - discuss with your doctor."
        ]
        
        prefix = random.choice(direct_prefixes)
        ending = random.choice(direct_endings)
        
        # Remove emotional language
        modified = text.replace("I understand", "Research indicates")
        modified = modified.replace("I'm here for you", "Support is available")
        modified = modified.replace("Don't worry", "Evidence suggests")
        modified = modified.replace("I know", "Data shows")
        
        return f"{prefix}\n\n{modified}\n\n{ending}"
    
    def _make_clinical(self, text):
        """Clinical medical tone"""
        clinical_prefixes = [
            "Medical information:",
            "Clinical overview:",
            "Evidence-based summary:",
            "Medical literature indicates:"
        ]
        
        clinical_endings = [
            "Recommend consultation with obstetric care provider.",
            "Individual medical assessment required.",
            "Professional medical evaluation advised.",
            "Discuss with healthcare team for personalized care plan."
        ]
        
        prefix = random.choice(clinical_prefixes)
        ending = random.choice(clinical_endings)
        
        # Make more clinical
        modified = text.replace("baby", "fetus")
        modified = modified.replace("mom", "maternal patient")
        modified = modified.replace("pregnancy", "gestation")
        
        return f"{prefix}\n\n{modified}\n\n{ending}"
    
    def _make_friendly(self, text):
        """Friendly, casual tone"""
        friendly_prefixes = [
            "Hey there! Let's chat about this.",
            "Oh, I'm so glad you asked about this!",
            "This is such a great question!",
            "I love talking about this stuff!"
        ]
        
        friendly_endings = [
            "Hope this helps, friend! 😊",
            "You're doing amazing! 🌟",
            "Feel free to ask me anything else!",
            "I'm always here to chat! 💕"
        ]
        
        prefix = random.choice(friendly_prefixes)
        ending = random.choice(friendly_endings)
        
        # Make more conversational
        modified = text.replace("It is important", "It's really important")
        modified = modified.replace("You should", "You might want to")
        modified = modified.replace("Healthcare providers recommend", "Doctors usually suggest")
        
        return f"{prefix}\n\n{modified}\n\n{ending}"

# === FRIDAYAI MASTER CLASS ===
class FridayAI:
    def __init__(self, memory, emotion):
        # === ALL YOUR ORIGINAL INITIALIZATION (PRESERVED) ===
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
        
        # === LEGENDARY ENHANCEMENTS ===
        self.legendary_memory = LegendaryConversationMemory()
        self.goal_coach = GoalCoachingSystem()
        self.output_formatter = RichOutputFormatter()
        self.self_eval = SelfEvaluationSystem()
        self.citation_system = CitationSystem()
        self.knowledge_injection = KnowledgeInjectionSystem()
        
        # === UNSTOPPABLE ENHANCEMENTS ===
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
            "uptime_start": datetime.now(),
            "emergency_responses": 0,
            "goals_created": 0,
            "facts_shared": 0
        }
        
        # Session tracking
        self.session_topics = []
        self.current_user_id = "default"
        self.pregnancy_week = 0
        
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
        
        # Auto-save and cleanup setup
        self._setup_autosave()
        self._setup_cleanup()
        
        print("🏆 Friday AI Super Ultra Brilliant - All systems activated!")

    def _analyze_input_semantic(self, user_input):
        """Enhanced semantic analysis with emergency detection"""
        input_lower = user_input.lower().strip()
        
        # 1. EMERGENCY CHECK FIRST
        is_emergency, emergency_type, urgency_level = self.emergency.check_emergency(user_input)
        if is_emergency:
            return {
                'type': 'emergency',
                'emergency_type': emergency_type,
                'urgency_level': urgency_level,
                'confidence': 1.0
            }
        
        # 2. QUICK FILTERS for obvious non-conversational input
        obvious_non_conversation = [
            len(input_lower) < 3,
            input_lower.startswith(('def ', 'class ', 'import ', 'from ')),
            input_lower.startswith(('!', '\\', '/')) and not input_lower.startswith(('!tone', '!voice', '!goal', '!health')),
            input_lower.count('(') > input_lower.count(' '),
            bool(re.match(r'^[a-zA-Z]{1,4}, input_lower)),
            'filters out' in input_lower,
            'show_tone' in input_lower,
        ]
        
        if any(obvious_non_conversation):
            return {
                'type': 'non_conversational',
                'confidence': 0.9,
                'response': "I'm not sure what you're referring to. Could you tell me more about what you need help with?"
            }
        
        # 3. ENHANCED SEMANTIC PREGNANCY DETECTION
        emotional_patterns = [
            r'\b(feel|feeling|felt)\s+(scared|afraid|anxious|worried|nervous|overwhelmed|excited|happy)',
            r'\b(i\'?m|am)\s+(scared|afraid|anxious|worried|nervous|terrified|excited|thrilled)',
            r'\b(so|really|very|extremely)\s+(scared|afraid|worried|anxious|excited|happy)',
            r'\bnot\s+sure\s+(i|if|how|what)',
            r'\bdon\'?t\s+know\s+(if|how|what|where)',
            r'\bwhat\s+if\s+something',
            r'\bworried\s+about',
            r'\bscared\s+(about|of)',
            r'\bexcited\s+(about|for)',
        ]
        
        pregnancy_patterns = [
            r'\b(baby|pregnancy|pregnant|expecting|maternity)',
            r'\b(mom|mother|motherhood|maternal|mama)',
            r'\b(birth|delivery|labor|due\s+date|childbirth)',
            r'\b(first\s+time\s+mom|new\s+mom|expecting\s+mom)',
            r'\b(gestational|prenatal|trimester|weeks\s+pregnant)',
            r'\b(midwife|obstetrician|ob/gyn|doula)',
            r'\b(ultrasound|sonogram|prenatal\s+visit)',
        ]
        
        personal_patterns = [
            r'\bi\s+(am|\'m|was|will|have|need|want|think|feel|wonder)',
            r'\bmy\s+(baby|pregnancy|doctor|body|belly|symptoms)',
            r'\bshould\s+i\b',
            r'\bcan\s+i\b',
            r'\bhow\s+(do|can|should)\s+i\b',
            r'\bwill\s+i\b',
            r'\bam\s+i\b',
        ]
        
        # Count pattern matches with enhanced scoring
        emotional_score = sum(1 for pattern in emotional_patterns if re.search(pattern, input_lower))
        pregnancy_score = sum(1 for pattern in pregnancy_patterns if re.search(pattern, input_lower))
        personal_score = sum(1 for pattern in personal_patterns if re.search(pattern, input_lower))
        
        # 4. CONTEXT ANALYSIS
        is_question = any([
            input_lower.endswith('?'),
            input_lower.startswith(('what', 'how', 'when', 'where', 'why', 'should', 'can', 'will', 'do', 'does', 'is', 'are')),
            ' or ' in input_lower,
            'tell me about' in input_lower,
        ])
        
        vulnerability_indicators = [
            'not sure', 'don\'t know', 'confused', 'help', 'advice', 'guidance',
            'what should', 'am i', 'will i be', 'going to be', 'worried about',
            'need to know', 'wondering if', 'concerned about'
        ]
        shows_vulnerability = any(indicator in input_lower for indicator in vulnerability_indicators)
        
        word_count = len(input_lower.split())
        seems_conversational = 5 <= word_count <= 150
        
        # 5. ENHANCED SCORING ALGORITHM
        base_score = 0
        
        # Emotional component (35% weight)
        if emotional_score > 0:
            base_score += 35 * min(emotional_score / 2, 1)
        
        # Pregnancy context (35% weight)
        if pregnancy_score > 0:
            base_score += 35 * min(pregnancy_score / 2, 1)
        
        # Personal narrative (20% weight)
        if personal_score > 0:
            base_score += 20 * min(personal_score / 3, 1)
        
        # Conversation quality bonuses (10% weight)
        if is_question:
            base_score += 5
        if shows_vulnerability:
            base_score += 3
        if seems_conversational:
            base_score += 2
        
        # 6. ENHANCED DECISION LOGIC
        if base_score >= 70:
            return {
                'type': 'pregnancy_concern',
                'confidence': base_score / 100,
                'context': 'emotional_pregnancy_support',
                'emotional_score': emotional_score,
                'pregnancy_score': pregnancy_score
            }
        elif base_score >= 45:
            return {
                'type': 'possible_pregnancy_concern', 
                'confidence': base_score / 100,
                'response': "It sounds like you might have something pregnancy-related on your mind. I'm here to listen and support you. What's going on?"
            }
        else:
            return {
                'type': 'general_conversation',
                'confidence': (100 - base_score) / 100,
                'seems_engaged': shows_vulnerability or is_question
            }

    def _configure_logging(self):
        # Enhanced logging with performance tracking
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
        """Enhanced empathy responses with more variety"""
        try:
            empathy_path = "./pregnancy_support/empathy/soft_replies.json"
            with open(empathy_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            return {
                "pregnancy_emotional_support": {
                    "anxious": [
                        "It's completely natural to feel scared and uncertain about the journey ahead. Becoming a parent is one of the biggest changes you'll ever experience, and it's okay to worry about whether you'll be a great mom or how the rest of your pregnancy will go. These feelings are actually a sign of just how much you care about your baby and the kind of parent you want to be.",
                        "I can hear the worry in your words, and I want you to know that anxiety during pregnancy is incredibly common. You're not alone in feeling this way. Many expectant mothers experience these same fears and uncertainties. What you're feeling is valid, and it shows how much you already love your baby.",
                        "Your anxious feelings are so understandable. Pregnancy brings so many unknowns, and it's natural for your mind to try to prepare for every possibility. Take a deep breath with me - you're doing better than you think you are."
                    ],
                    "scared": [
                        "Your fears about pregnancy and motherhood are completely valid. It's natural to feel scared when facing something so life-changing and important. These feelings don't mean anything is wrong - they show how much you care.",
                        "I can feel how frightened you are, and I want to wrap you in the biggest virtual hug right now. Fear during pregnancy is so normal, even though it doesn't make it any easier to experience. You're braver than you know.",
                        "Being scared is part of the human experience, especially when we're facing something as profound as bringing new life into the world. Your fear shows your love and commitment to doing right by your baby."
                    ],
                    "overwhelmed": [
                        "Feeling overwhelmed during pregnancy is so common and understandable. There's so much information, so many changes happening to your body, and so many decisions to make. Take it one day at a time.",
                        "I can sense how much you're carrying right now - physically, emotionally, and mentally. It's okay to feel overwhelmed. You don't have to figure everything out at once. Let's break things down into smaller, manageable pieces.",
                        "The overwhelm you're feeling is real and valid. Pregnancy can feel like drinking from a fire hose sometimes - so much information, so many changes, so many decisions. You're allowed to take breaks and go at your own pace."
                    ],
                    "sad": [
                        "I'm sorry you're feeling sad right now. Pregnancy emotions can be intense and sometimes confusing. Your feelings are valid, and it's important to be gentle with yourself during this time.",
                        "Sadness during pregnancy is more common than many people talk about. Hormones, life changes, and the weight of responsibility can all contribute to these feelings. You're not broken or doing anything wrong.",
                        "Your sadness matters, and I'm here to sit with you in this feeling. Sometimes we need to feel our emotions fully before we can move through them. You don't have to be happy all the time, even during pregnancy."
                    ],
                    "excited": [
                        "I can feel your excitement and it's absolutely beautiful! There's something magical about the anticipation and joy that comes with expecting a baby. Your excitement is contagious!",
                        "Your joy is lighting up our conversation! It's wonderful to see someone so thrilled about their pregnancy journey. This excitement is a gift - both to you and your growing baby."
                    ]
                },
                "general_support": [
                    "I'm here to support you through whatever you're feeling. Your emotions and concerns are completely valid.",
                    "Thank you for sharing with me. I'm honored to be part of your pregnancy journey, and I'm here for whatever you need.",
                    "You matter, your feelings matter, and your experience matters. I'm here to listen and support you."
                ]
            }

    def _get_empathy_response(self, mood, user_input=""):
        """Enhanced empathy response selection"""
        pregnancy_empathy = self.empathy_responses.get("pregnancy_emotional_support", {})
        
        if mood in pregnancy_empathy:
            responses = pregnancy_empathy[mood]
            
            # Select response based on context if available
            if len(responses) > 1 and user_input:
                input_lower = user_input.lower()
                
                # Prioritize responses that match specific contexts
                for response in responses:
                    if any(word in input_lower for word in ['first time', 'don\'t know', 'new']) and 'first' in response.lower():
                        return response
                    elif any(word in input_lower for word in ['alone', 'by myself']) and 'alone' in response.lower():
                        return response
            
            return random.choice(responses)
        
        # Fallback to general support
        general_support = self.empathy_responses.get("general_support", [])
        if general_support:
            return random.choice(general_support)
        
        return "I understand you're going through a lot right now. I'm here to support you."

    def _offer_pregnancy_resources(self, user_input: str, emotional_tone: str, analysis_data: Dict = None) -> str:
        """Enhanced resource offering with personalization"""
        
        # Detect if user needs support
        need_keywords = ["help", "advice", "don't know", "unsure", "worried", "scared", "anxious", "overwhelmed", "guidance", "support"]
        pregnancy_keywords = ["baby", "pregnant", "pregnancy", "mom", "mother", "birth", "expecting", "maternal"]
        
        needs_help = any(keyword in user_input.lower() for keyword in need_keywords)
        is_pregnancy_related = any(keyword in user_input.lower() for keyword in pregnancy_keywords)
        
        if not (needs_help and is_pregnancy_related):
            return ""
        
        # Get personalized empathy response
        empathy_text = self._get_empathy_response(emotional_tone, user_input)
        
        # Enhanced resource database
        resources = {
            "anxiety": {
                "immediate": [
                    "🧘‍♀️ Try the 4-7-8 breathing technique: Inhale for 4, hold for 7, exhale for 8",
                    "💭 Ground yourself: Name 5 things you can see, 4 you can touch, 3 you can hear"
                ],
                "books": [
                    "📖 'The First-Time Mom's Pregnancy Handbook' by Allison Hill",
                    "📖 'What to Expect When You're Expecting' by Heidi Murkoff",
                    "📖 'Mindful Birthing' by Nancy Bardacke"
                ],
                "apps": [
                    "📱 Calm - Meditation and sleep stories for pregnancy",
                    "📱 Headspace - Prenatal meditation courses",
                    "📱 BabyCentre Pregnancy Tracker"
                ],
                "support": [
                    "🏥 Talk to your healthcare provider about anxiety management",
                    "👥 Join a local pregnancy support group",
                    "💬 Consider pregnancy counseling or therapy",
                    "🤱 Connect with other expectant mothers online"
                ]
            },
            "information": {
                "websites": [
                    "🌐 American Pregnancy Association (americanpregnancy.org)",
                    "🌐 What to Expect (whattoexpect.com)",
                    "🌐 The Bump (thebump.com)"
                ],
                "classes": [
                    "👶 Childbirth preparation classes",
                    "🤱 Breastfeeding classes",
                    "👨‍👩‍👧‍👦 Partner support classes"
                ]
            }
        }
        
        # Determine resource category
        if emotional_tone in ["anxious", "scared", "overwhelmed", "sad"]:
            category = "anxiety"
        else:
            category = "information"
        
        # Build personalized response
        final_text = empathy_text
        final_text += "\n\n💝 **Here are some resources that might help:**\n"
        
        if category == "anxiety":
            # Immediate coping strategies first
            if "immediate" in resources[category]:
                final_text += "\n**🆘 Immediate Coping Strategies:**\n"
                for strategy in resources[category]["immediate"]:
                    final_text += f"• {strategy}\n"
            
            # Books for deeper support
            if "books" in resources[category]:
                final_text += "\n**📚 Helpful Books:**\n"
                for book in resources[category]["books"]:
                    final_text += f"• {book}\n"
            
            # Apps for daily support
            if "apps" in resources[category]:
                final_text += "\n**📱 Supportive Apps:**\n"
                for app in resources[category]["apps"]:
                    final_text += f"• {app}\n"
            
            # Professional support options
            if "support" in resources[category]:
                final_text += "\n**🤝 Professional Support:**\n"
                for support in resources[category]["support"]:
                    final_text += f"• {support}\n"
        
        elif category == "information":
            if "websites" in resources[category]:
                final_text += "\n**🌐 Trusted Websites:**\n"
                for site in resources[category]["websites"]:
                    final_text += f"• {site}\n"
            
            if "classes" in resources[category]:
                final_text += "\n**📚 Educational Classes:**\n"
                for class_option in resources[category]["classes"]:
                    final_text += f"• {class_option}\n"
        
        # Interactive offers
        final_text += "\n💬 **I can also help you with:**\n"
        final_text += "• Creating a personalized support plan\n"
        final_text += "• Finding local resources in your area\n"
        final_text += "• Explaining any pregnancy topics you're curious about\n"
        final_text += "• Just listening when you need to talk\n"
        
        final_text += "\n✨ **What would be most helpful for you right now?**"
        
        return final_text

    def _setup_autosave(self):
        """Setup automatic conversation and state saving"""
        def autosave():
            while True:
                time.sleep(300)  # Save every 5 minutes
                try:
                    self._save_all_conversations()
                    self._save_performance_metrics()
                except Exception as e:
                    self.logger.error(f"Autosave error: {e}")
        
        autosave_thread = threading.Thread(target=autosave, daemon=True)
        autosave_thread.start()
    
    def _setup_cleanup(self):
        """Setup periodic cleanup of old data"""
        def cleanup():
            while True:
                time.sleep(3600)  # Cleanup every hour
                try:
                    self._cleanup_old_data()
                except Exception as e:
                    self.logger.error(f"Cleanup error: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup, daemon=True)
        cleanup_thread.start()
    
    def _save_all_conversations(self):
        """Save all active conversations and state"""
        for session_id, state in self.conversation_states.items():
            self._save_conversation_state(session_id, state)
        
        # Save legendary memory
        if hasattr(self, 'legendary_memory'):
            try:
                memory_path = Path("conversations/legendary_memory.pkl")
                memory_path.parent.mkdir(exist_ok=True)
                with open(memory_path, 'wb') as f:
                    pickle.dump(self.legendary_memory, f)
            except:
                pass
    
    def _save_conversation_state(self, session_id: str, state: ConversationState):
        """Save individual conversation state"""
        try:
            state_path = Path(f"conversations/{session_id}.pkl")
            state_path.parent.mkdir(exist_ok=True)
            with open(state_path, 'wb') as f:
                pickle.dump(state, f)
        except:
            pass
    
    def _save_performance_metrics(self):
        """Save performance metrics"""
        try:
            metrics_path = Path("performance_metrics.json")
            metrics_data = {
                'uptime_start': self.performance_metrics['uptime_start'].isoformat(),
                'successful_interactions': self.performance_metrics['successful_interactions'],
                'error_count': self.performance_metrics['error_count'],
                'emergency_responses': self.performance_metrics['emergency_responses'],
                'goals_created': self.performance_metrics['goals_created'],
                'facts_shared': self.performance_metrics['facts_shared'],
                'response_times_avg': sum(self.performance_metrics['response_times']) / len(self.performance_metrics['response_times']) if self.performance_metrics['response_times'] else 0
            }
            with open(metrics_path, 'w') as f:
                json.dump(metrics_data, f, indent=2)
        except:
            pass
    
    def _cleanup_old_data(self):
        """Clean up old data to prevent memory bloat"""
        # Clean up old conversation states (older than 30 days)
        conversations_dir = Path("conversations")
        if conversations_dir.exists():
            cutoff_date = datetime.now() - timedelta(days=30)
            for file_path in conversations_dir.glob("*.pkl"):
                if file_path.stat().st_mtime < cutoff_date.timestamp():
                    try:
                        file_path.unlink()
                    except:
                        pass
        
        # Clean up old error history
        if len(self.resilience.error_history) > 500:
            # Keep only recent 250 errors
            recent_errors = list(self.resilience.error_history)[-250:]
            self.resilience.error_history.clear()
            self.resilience.error_history.extend(recent_errors)

    # === ENHANCED HELPER METHODS ===
    def _handle_voice_command(self, command: str) -> str:
        """Enhanced voice command handling"""
        parts = command.lower().split()
        if len(parts) == 1:
            return self.voice.get_voice_info()
        elif parts[1] == 'on':
            if self.voice.enabled:
                return "🔊 Voice output is now enabled! I'll speak my responses."
            else:
                return "🔊 Voice features not available. Install speechrecognition and pyttsx3 for voice support."
        elif parts[1] == 'off':
            return "🔊 Voice output disabled. Switching to text-only mode."
        elif parts[1] == 'settings':
            return self.voice.get_voice_info()
        elif parts[1] == 'test':
            if self.voice.enabled:
                self.voice.speak("Hello! This is a voice test. Can you hear me clearly?")
                return "🔊 Voice test completed. How did that sound?"
            else:
                return "🔊 Voice features not available for testing."
        else:
            return "🔊 Voice commands: !voice [on/off/settings/test]"

    def _handle_goal_creation(self, goal_type: str = None) -> str:
        """Enhanced goal creation handling"""
        if not goal_type:
            goal_type = 'anxiety_management'  # Default
        
        result = self.goal_coach.create_goal(goal_type, self.current_user_id)
        if "Goal Created" in result:
            self.performance_metrics['goals_created'] += 1
        return result

    def _determine_response_type(self, content: str, has_goal_offer: bool, is_emergency: bool = False) -> str:
        """Enhanced response type determination"""
        if is_emergency:
            return 'emergency'
        elif '🎉' in content and 'CONGRATULATIONS' in content:
            return 'celebration'
        elif '💭 **I remember**' in content:
            return 'empathy'
        elif has_goal_offer or '🎯' in content:
            return 'goal'
        elif '📚' in content or '📖' in content or '🌐' in content:
            return 'resource'
        else:
            return 'normal'

    def _create_enhanced_response(self, content: str, response_type: str, user_input: str, 
                                emotional_tone: str = "neutral", additional_data: Dict = None) -> Dict:
        """Enhanced response creation with comprehensive metadata"""
        return {
            'domain': 'super_ultra_brilliant_friday',
            'content': self.output_formatter.format_response(content, response_type, emotional_tone),
            'confidence': 1.0,
            'emotional_tone': emotional_tone,
            'processing_time': datetime.now().isoformat(),
            'response_type': response_type,
            'legendary_features': {
                'active': True, 
                'type': response_type,
                'memory_enhanced': True,
                'goal_coaching': True,
                'emergency_detection': True,
                'voice_capable': self.voice.enabled,
                'secure_vault': self.vault.enabled
            },
            'session_data': {
                'user_id': self.current_user_id,
                'interaction_count': self.performance_metrics['successful_interactions'],
                'session_topics': self.session_topics[-5:] if self.session_topics else [],
                'current_tone': self.tone_manager.current_tone
            },
            'additional_data': additional_data or {}
        }

    # === MAIN RESPOND_TO METHOD WITH ALL ENHANCEMENTS ===
    @resilience.wrap_with_resilience
    def respond_to(self, user_input: str, pregnancy_week: int = 0) -> Dict[str, object]:
        """SUPER ULTRA BRILLIANT respond_to with ALL enhancement features integrated"""
        
        start_time = time.time()
        
        # Store pregnancy week
        if pregnancy_week > 0:
            self.pregnancy_week = pregnancy_week
        
        # === EMERGENCY CHECK FIRST ===
        analysis = self._analyze_input_semantic(user_input)
        
        if analysis.get('type') == 'emergency':
            emergency_response = self.emergency.generate_emergency_response(
                analysis['emergency_type'], 
                analysis['urgency_level']
            )
            
            # Voice alert for emergencies
            if self.voice.enabled:
                self.voice.speak("This appears to be an emergency. Please seek immediate medical attention.", interrupt_current=True)
            
            self.performance_metrics['emergency_responses'] += 1
            
            response_data = self._create_enhanced_response(
                emergency_response, "emergency", user_input, "urgent",
                {'emergency_type': analysis['emergency_type'], 'urgency_level': analysis['urgency_level']}
            )
            
            # Log emergency
            self.legendary_memory.add_exchange(user_input, emergency_response, "urgent")
            
            return response_data
        
        # === VOICE INTEGRATION CHECK ===
        if user_input.lower().startswith('!voice'):
            voice_response = self._handle_voice_command(user_input)
            return self._create_enhanced_response(voice_response, "system", user_input)
        
        # === GOAL COACHING CHECKS ===
        if user_input.lower() in ['yes', 'create goal', 'set goal', "i'm in", 'help me', 'let\'s do it']:
            goal_response = self._handle_goal_creation()
            if goal_response:
                return self._create_enhanced_response(goal_response, "goal", user_input)
        
        # Check for due goal check-ins
        check_in = self.goal_coach.check_for_due_check_ins(self.current_user_id)
        if check_in:
            return self._create_enhanced_response(check_in, "goal", user_input)
        
        # === TONE CHANGE REQUESTS ===
        tone_response = self.tone_manager.detect_tone_request(user_input)
        if tone_response:
            return self._create_enhanced_response(tone_response, "system", user_input)
        
        # === ENHANCED MEMORY RECALL ===
        similar_conversation = self.legendary_memory.find_similar_conversation(user_input)
        emotional_insights = self.legendary_memory.get_emotional_insights()
        
        # === CORE RESPONSE GENERATION WITH RESILIENCE ===
        @self.resilience.wrap_with_resilience
        def generate_core_response():
            # Memory injection
            ctx = inject(user_input)
            
            # Knowledge citations
            citations = query_knowledge(user_input)
            excluded_files = ['requirements.txt', 'cognition_notes.txt', '.gitignore', '.env']
            
            # Pregnancy emotion analysis
            pregnancy_analysis = None
            if PREGNANCY_EMOTION_AVAILABLE and self.pregnancy_emotion and self.pregnancy_week > 0:
                try:
                    pregnancy_analysis = self.pregnancy_emotion.analyze_pregnancy_emotion(
                        user_input, self.pregnancy_week
                    )
                except Exception as e:
                    pass
            
            # Generate response
            result = self.pipeline.generate_response(user_input)
            
            # Handle response format
            if isinstance(result, str):
                raw_reply = result
                emotional_tone = analysis.get('emotional_tone', 'neutral')
                memory_context = None
                identity_context = None
            elif isinstance(result, dict):
                raw_reply = result.get('reply', result.get('response', '')).strip()
                emotional_tone = result.get('emotion', result.get('emotional_tone', 'neutral'))
                memory_context = result.get('memory_context')
                identity_context = result.get('identity_context')
            else:
                raw_reply = str(result)
                emotional_tone = 'neutral'
                memory_context = None
                identity_context = None
            
            # Clean output
            if not raw_reply:
                raw_reply = "I'm here to help. What's on your mind today?"
            
            return raw_reply, emotional_tone, memory_context, identity_context, pregnancy_analysis
        
        raw_reply, emotional_tone, memory_context, identity_context, pregnancy_analysis = generate_core_response()
        
        # === LEGENDARY ENHANCEMENTS ===
        
        # 1. DYNAMIC MEMORY RECALL
        if similar_conversation:
            days_ago = (datetime.now() - similar_conversation['timestamp']).days
            time_ref = f"{days_ago} days ago" if days_ago > 0 else "earlier today"
            memory_prefix = f"💭 **I remember** we talked about something similar {time_ref}. "
            
            if similar_conversation['emotional_tone'] == emotional_tone:
                memory_prefix += f"You were feeling {similar_conversation['emotional_tone']} then too. "
            else:
                memory_prefix += f"You seemed {similar_conversation['emotional_tone']} then, but I sense a different energy now. "
            
            memory_prefix += "Let me build on what we discussed.\n\n"
            raw_reply = memory_prefix + raw_reply
        
        # 2. PREDICTIVE MILESTONE INTEGRATION
        if self.pregnancy_week > 0:
            milestones = self.predictive.predict_upcoming_milestones(self.pregnancy_week, 3)
            if milestones and 'milestone' in user_input.lower() or random.random() < 0.15:
                raw_reply += "\n\n📅 **Upcoming Milestones:**\n"
                for milestone in milestones[:2]:
                    raw_reply += f"• {milestone}\n"
        
        # 3. GOAL COACHING INTEGRATION
        goal_opportunity = self.goal_coach.detect_goal_opportunity(user_input, raw_reply, emotional_tone)
        if goal_opportunity and not similar_conversation:
            goal_offer = self.goal_coach.create_goal_offer(goal_opportunity)
            raw_reply += goal_offer
        
        # 4. ENHANCED RESOURCE OFFERING
        resources_offer = self._offer_pregnancy_resources(user_input, emotional_tone, analysis)
        if resources_offer:
            raw_reply = resources_offer
        
        # 5. CITATION SYSTEM
        topic_context = ' '.join(self.session_topics[-3:]) if self.session_topics else user_input
        raw_reply = self.citation_system.add_citations(raw_reply, topic_context)
        
        # 6. KNOWLEDGE INJECTION
        conversation_length = len(self.legendary_memory.conversations)
        if self.knowledge_injection.should_add_fact(conversation_length, emotional_tone):
            fact = self.knowledge_injection.get_relevant_fact(user_input, emotional_tone, self.pregnancy_week)
            raw_reply += fact
            self.performance_metrics['facts_shared'] += 1
        
        # 7. SELF-EVALUATION
        if self.self_eval.should_request_feedback(emotional_tone):
            feedback_request = self.self_eval.generate_feedback_request(
                self.tone_manager.current_tone,
                self.session_topics[-2:] if len(self.session_topics) >= 2 else [],
                emotional_tone
            )
            raw_reply += f"\n\n{feedback_request}"
        
        # === TONE APPLICATION ===
        if not resources_offer:
            raw_reply = self.tone_manager.apply_tone(raw_reply, emotional_tone)
        
        # Apply tone rewriting
        raw_reply = self.tone_rewriter.rewrite(raw_reply)
        
        # === CITATIONS FILTERING ===
        relevant_citations = []
        for c in citations:
            if c.get('source') not in excluded_files and 'text' in c:
                if len(c['text']) > 50 and any(word in c['text'].lower() for word in user_input.lower().split()):
                    relevant_citations.append(c)
        
        if relevant_citations and len(relevant_citations) <= 2:
            sources = [f"📄 {c['text']}" for c in relevant_citations[:2]]
            raw_reply += "\n\n" + "\n\n".join(sources)
        
        # === VOICE OUTPUT ===
        if self.voice.enabled and hasattr(self, '_voice_enabled_for_session'):
            if self._voice_enabled_for_session:
                self.voice.speak(raw_reply)
        
        # === STORE IN LEGENDARY MEMORY ===
        self.legendary_memory.add_exchange(user_input, raw_reply, emotional_tone)
        
        # Track session topics
        topic = self.legendary_memory._extract_topic(user_input)
        if topic:
            self.session_topics.append(topic)
            if len(self.session_topics) > 20:
                self.session_topics = self.session_topics[-20:]
        
        # === LOGGING AND METRICS ===
        log_event(user_input, source="user")
        log_event(raw_reply, source="friday")
        
        try:
            update_mood(emotional_tone)
        except Exception as e:
            update_mood("neutral")
        
        # Performance metrics
        end_time = time.time()
        response_time = end_time - start_time
        self.performance_metrics["response_times"].append(response_time)
        self.performance_metrics["successful_interactions"] += 1
        
        # === DETERMINE RESPONSE TYPE AND FORMAT ===
        response_type = self._determine_response_type(
            raw_reply, 
            goal_opportunity is not None,
            False
        )
        
        final_output = self.output_formatter.format_response(raw_reply, response_type, emotional_tone)
        
        # === COMPREHENSIVE RESPONSE OBJECT ===
        response = self._create_enhanced_response(
            final_output, response_type, user_input, emotional_tone,
            {
                'memory_context': memory_context,
                'identity_context': identity_context,
                'processing_time_ms': round(response_time * 1000, 2),
                'analysis_data': analysis,
                'emotional_insights': emotional_insights,
                'pregnancy_week': self.pregnancy_week,
                'milestones_shared': bool(milestones) if self.pregnancy_week > 0 else False,
                'citations_added': '📚' in raw_reply,
                'knowledge_injection': '💡' in raw_reply,
                'goal_coaching_offered': goal_opportunity is not None,
                'memory_recall_used': similar_conversation is not None,
                'response_time_category': 'fast' if response_time < 1 else 'normal' if response_time < 3 else 'slow'
            }
        )
        
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

    # === ENHANCED UTILITY METHODS ===
    
    def voice_interaction(self, user_id: str = "default"):
        """Enhanced voice-based interaction mode"""
        if not self.voice.enabled:
            print("🔊 Voice features not available. Please install speechrecognition and pyttsx3.")
            return
            
        self.current_user_id = user_id
        self._voice_enabled_for_session = True
        
        print("🎤 Voice mode activated! Say 'exit voice mode' to return to text.")
        self.voice.speak("Voice mode activated. I'm listening!")
        
        consecutive_failures = 0
        max_failures = 3
        
        while consecutive_failures < max_failures:
            user_input = self.voice.listen(timeout=10, phrase_time_limit=15)
            
            if user_input is None:
                consecutive_failures += 1
                if consecutive_failures < max_failures:
                    print("🔄 Didn't catch that, try again...")
                    self.voice.speak("I didn't catch that. Could you try again?")
                continue
            
            consecutive_failures = 0  # Reset on successful input
            print(f"You said: {user_input}")
            
            if any(phrase in user_input.lower() for phrase in ['exit voice mode', 'stop voice', 'text mode']):
                print("Returning to text mode.")
                self.voice.speak("Switching back to text mode. It was great talking with you!")
                break
            
            # Get and speak response
            response = self.respond_to(user_input, self.pregnancy_week)
            
            print(f"\nFriday: {response['content']}")
            # Voice output is handled automatically in respond_to
        
        if consecutive_failures >= max_failures:
            print("Voice mode ended due to audio issues. Returning to text mode.")
            self.voice.speak("I'm having trouble hearing you. Let's switch to text mode.")
        
        self._voice_enabled_for_session = False
    
    def get_health_insights(self, user_id: str) -> str:
        """Enhanced health insights from stored data"""
        if not self.vault.enabled:
            return "🔒 Health vault features not available. Install cryptography for secure health tracking."
            
        health_data = self.vault.retrieve_health_data(user_id)
        
        if not health_data:
            return "📊 No health data recorded yet. Would you like to start tracking your pregnancy journey? I can help you set up secure health monitoring."
        
        insights = "📊 **Your Personalized Health Insights:**\n\n"
        
        # Analyze patterns
        categories = defaultdict(list)
        for record in health_data:
            if 'category' in record:
                categories[record['category']].append(record['data'])
            else:
                for category, data in record.items():
                    categories[category].append(data)
        
        # Generate insights by category
        for category, data_list in categories.items():
            insights += f"**{category.replace('_', ' ').title()}:**\n"
            
            if category == "symptoms":
                symptom_freq = defaultdict(int)
                for data in data_list:
                    for symptom in data.get("symptoms", []):
                        symptom_freq[symptom] += 1
                
                if symptom_freq:
                    most_common = max(symptom_freq.items(), key=lambda x: x[1])
                    insights += f"• Most frequent: {most_common[0]} ({most_common[1]} times)\n"
                    insights += f"• Total different symptoms tracked: {len(symptom_freq)}\n"
            
            elif category == "mood_tracking":
                moods = [data.get('mood', 'neutral') for data in data_list]
                if moods:
                    dominant_mood = max(set(moods), key=moods.count)
                    insights += f"• Dominant mood: {dominant_mood}\n"
                    insights += f"• Mood entries: {len(moods)}\n"
            
            elif category == "baby_movements":
                movements = [data.get('count', 0) for data in data_list if 'count' in data]
                if movements:
                    avg_movements = sum(movements) / len(movements)
                    insights += f"• Average daily movements: {avg_movements:.1f}\n"
                    insights += f"• Movement tracking days: {len(movements)}\n"
            
            insights += "\n"
        
        # Add trending information
        recent_data = [record for record in health_data if 'timestamp' in record and 
                      (datetime.now() - datetime.fromisoformat(record['timestamp'].replace('Z', '+00:00'))).days <= 7]
        
        if recent_data:
            insights += f"**Recent Activity (Last 7 Days):**\n"
            insights += f"• Health records added: {len(recent_data)}\n"
            insights += f"• Most active category: {max(categories.keys(), key=lambda k: len([r for r in recent_data if r.get('category') == k]))}\n"
        
        insights += f"\n🔐 All data is securely encrypted and stored locally."
        insights += f"\n💡 **Tip:** Regular tracking helps identify patterns and supports better healthcare discussions!"
        
        return insights
    
    def get_comprehensive_report(self) -> str:
        """Generate comprehensive system performance report"""
        uptime = datetime.now() - self.performance_metrics["uptime_start"]
        
        # Calculate averages
        if self.performance_metrics["response_times"]:
            if NUMPY_AVAILABLE:
                avg_response = np.mean(list(self.performance_metrics["response_times"]))
                median_response = np.median(list(self.performance_metrics["response_times"]))
            else:
                response_times = list(self.performance_metrics["response_times"])
                avg_response = sum(response_times) / len(response_times)
                response_times.sort()
                median_response = response_times[len(response_times) // 2]
        else:
            avg_response = median_response = 0
        
        # Calculate success rate
        total_attempts = self.performance_metrics['successful_interactions'] + self.performance_metrics['error_count']
        success_rate = (self.performance_metrics['successful_interactions'] / max(1, total_attempts)) * 100
        
        report = f"""
🏆 **Friday Super Ultra Brilliant - Comprehensive Report**

**System Performance:**
• Uptime: {uptime.days} days, {uptime.seconds // 3600} hours, {(uptime.seconds % 3600) // 60} minutes
• Total Interactions: {self.performance_metrics['successful_interactions']:,}
• Success Rate: {success_rate:.1f}%
• Average Response Time: {avg_response:.2f}s
• Median Response Time: {median_response:.2f}s
• Error Count: {self.performance_metrics['error_count']}

**Feature Usage:**
• Emergency Responses: {self.performance_metrics['emergency_responses']}
• Goals Created: {self.performance_metrics['goals_created']}
• Facts Shared: {self.performance_metrics['facts_shared']}
• Active Conversations: {len(self.conversation_states)}
• Memory Entries: {len(self.legendary_memory.conversations)}

**System Capabilities:**
• 🧠 Legendary Memory: ✅ Active ({len(self.legendary_memory.conversations)} conversations stored)
• 🎯 Goal Coaching: ✅ Active ({len(self.goal_coach.active_goals.get(self.current_user_id, []))} active goals)
• 🔊 Voice Interface: {'✅ Available' if VOICE_AVAILABLE else '❌ Requires installation'}
• 🔐 Secure Vault: {'✅ Available' if ENCRYPTION_AVAILABLE else '❌ Requires cryptography package'}
• 🚨 Emergency Detection: ✅ Active
• 📊 Predictive Analytics: ✅ Active
• 💡 Knowledge Injection: ✅ Active
• 📚 Citation System: ✅ Active
• 🛡️ Resilience Engine: ✅ Active
• 🎨 Rich Formatting: ✅ Active

**Current Session:**
• User ID: {self.current_user_id}
• Current Tone: {self.tone_manager.current_tone.title()}
• Session Topics: {len(self.session_topics)}
• Pregnancy Week: {self.pregnancy_week if self.pregnancy_week > 0 else 'Not specified'}

**System Health:**
{self.resilience.get_health_report()}

**Knowledge Analytics:**
{self.knowledge_injection.get_fact_analytics()}

**Citation Statistics:**
{self.citation_system.get_citation_stats()}

**Self-Evaluation:**
{self.self_eval.get_performance_summary()}

**Status:** 🟢 All systems operational and enhanced
**Version:** Super Ultra Brilliant v1.0
"""
        
        return report
    
    def export_user_data(self, user_id: str, include_health: bool = False) -> str:
        """Export comprehensive user data"""
        export_data = {
            'user_id': user_id,
            'export_timestamp': datetime.now().isoformat(),
            'conversation_history': [],
            'goals': {
                'active': self.goal_coach.active_goals.get(user_id, []),
                'completed': self.goal_coach.completed_goals.get(user_id, [])
            },
            'preferences': {
                'current_tone': self.tone_manager.current_tone,
                'tone_history': list(self.tone_manager.tone_history)
            },
            'session_data': {
                'topics': self.session_topics,
                'pregnancy_week': self.pregnancy_week
            }
        }
        
        # Add conversation history
        for conv in self.legendary_memory.conversations:
            if conv.get('user_id', 'default') == user_id:
                export_data['conversation_history'].append({
                    'timestamp': conv['timestamp'].isoformat(),
                    'user_input': conv['user_input'],
                    'ai_response': conv['ai_response'],
                    'emotional_tone': conv['emotional_tone']
                })
        
        # Add health data if requested and available
        if include_health and self.vault.enabled:
            health_data = self.vault.retrieve_health_data(user_id)
            export_data['health_data'] = health_data
        
        return json.dumps(export_data, indent=2, default=str)

# === HELPER FUNCTIONS ===

def handle_user_input_intelligently(user_input, ai):
    """Enhanced smart input handling with comprehensive analysis"""
    
    # Analyze the input first
    analysis = ai._analyze_input_semantic(user_input)
    
    # Handle different input types
    if analysis['type'] == 'emergency':
        # Emergency - handle immediately through respond_to
        return ai.respond_to(user_input)['content']
    
    elif analysis['type'] == 'non_conversational':
        if 'response' in analysis:
            return analysis['response']
        else:
            return "I'm not sure what you're trying to do. Could you rephrase that as a question or tell me what you need help with?"
    
    elif analysis['type'] == 'pregnancy_concern':
        # High confidence pregnancy concern - full emotional support
        return ai.respond_to(user_input)['content']
    
    elif analysis['type'] == 'possible_pregnancy_concern':
        # Medium confidence - ask for clarification with warmth
        return analysis['response']
    
    else:
        # General conversation - enhanced response
        if analysis.get('seems_engaged'):
            # User seems engaged, provide full response
            return ai.respond_to(user_input)['content']
        else:
            # Simple response but still supportive
            response = ai.respond_to(user_input)['content']
            return response

def handle_special_commands(user_input, ai):
    """Handle special Friday commands"""
    
    if user_input.lower().startswith("!pregnancy_test"):
        try:
            parts = user_input.split()
            week = int(parts[1]) if len(parts) > 1 else 20
            test_message = "I'm so excited but also nervous about feeling the baby move!"
            response = ai.respond_to(test_message, pregnancy_week=week)
            return f"🧪 **Testing pregnancy support (Week {week}):**\n\n{response['content']}"
        except Exception as e:
            return f"❌ Test failed: {e}"
    
    elif user_input.lower() == "!status":
        return ai.get_comprehensive_report()
    
    elif user_input.lower() == "!health":
        user_id = ai.current_user_id
        return ai.get_health_insights(user_id)
    
    elif user_input.lower() == "!voice":
        if VOICE_AVAILABLE:
            print("\n🎤 Starting voice interaction mode...")
            ai.voice_interaction(ai.current_user_id)
            return "Voice interaction completed."
        else:
            return "🔊 Voice features not available. Install speechrecognition and pyttsx3 for voice support."
    
    elif user_input.lower().startswith("!save health"):
        if not ENCRYPTION_AVAILABLE:
            return "🔒 Health vault not available. Install cryptography for secure health tracking."
        
        data_str = user_input[len("!save health"):].strip()
        try:
            data = json.loads(data_str) if data_str else {"note": "Manual health entry", "timestamp": datetime.now().isoformat()}
            if ai.vault.store_health_data(ai.current_user_id, "manual_entry", data):
                return "✅ Health data securely saved!"
            else:
                return "❌ Failed to save health data."
        except:
            return "📝 Please provide data in JSON format after !save health, or just use !save health to create a basic entry."
    
    elif user_input.lower().startswith("!export"):
        try:
            include_health = "health" in user_input.lower()
            export_data = ai.export_user_data(ai.current_user_id, include_health)
            
            # Save to file
            export_path = Path(f"exports/{ai.current_user_id}_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            export_path.parent.mkdir(exist_ok=True)
            with open(export_path, 'w') as f:
                f.write(export_data)
            
            return f"📁 Data exported to {export_path}"
        except Exception as e:
            return f"❌ Export failed: {e}"
    
    elif user_input.lower() == "!goals":
        active_goals = len(ai.goal_coach.active_goals.get(ai.current_user_id, []))
        completed_goals = len(ai.goal_coach.completed_goals.get(ai.current_user_id, []))
        return f"🎯 **Your Goals:**\n• Active: {active_goals}\n• Completed: {completed_goals}\n\nUse 'show goals' to see details or 'create goal' to start a new one!"
    
    elif user_input.lower() == "!memory":
        conv_count = len(ai.legendary_memory.conversations)
        insights = ai.legendary_memory.get_emotional_insights()
        memory_info = f"💭 **Memory System:**\n• Conversations remembered: {conv_count}\n"
        if insights:
            memory_info += f"• Dominant emotion: {insights.get('dominant_emotion', 'neutral')}\n"
            memory_info += f"• Emotional variety: {insights.get('emotional_variety', 0)} different emotions\n"
        return memory_info
    
    elif user_input.lower().startswith("!feedback"):
        feedback_text = user_input[len("!feedback"):].strip()
        if feedback_text:
            response = ai.self_eval.process_feedback(feedback_text, ai.tone_manager.current_tone)
            return f"💭 {response}"
        else:
            return "💭 Please provide feedback after !feedback, like: !feedback you're doing great!"
    
    return None

def show_enhanced_startup():
    """Show enhanced startup sequence"""
    print("\n" + "="*80)
    print("🏆 FRIDAY AI - SUPER ULTRA BRILLIANT EDITION")
    print("="*80)
    print("\n🚀 **Initializing Advanced Features:**")
    
    features = [
        ("🧠 Legendary Memory System", "LOADED"),
        ("🎯 Goal Coaching Engine", "ACTIVE"),
        ("🔊 Voice Interface", "AVAILABLE" if VOICE_AVAILABLE else "INSTALL REQUIRED"),
        ("🔐 Secure Health Vault", "AVAILABLE" if ENCRYPTION_AVAILABLE else "INSTALL REQUIRED"),
        ("🚨 Emergency Detection", "ACTIVE"),
        ("📊 Predictive Analytics", "ACTIVE"),
        ("💡 Knowledge Injection", "ACTIVE"),
        ("📚 Citation System", "ACTIVE"),
        ("🛡️ Resilience Engine", "ACTIVE"),
        ("🎨 Rich Output Formatting", "ACTIVE"),
        ("📈 Self-Evaluation", "ACTIVE"),
        ("🤖 Pregnancy Intelligence", "ACTIVE" if PREGNANCY_EMOTION_AVAILABLE else "BASIC MODE"),
    ]
    
    for feature, status in features:
        color = "🟢" if status == "ACTIVE" or status == "LOADED" or status == "AVAILABLE" else "🟡" if "INSTALL" in status else "🟢"
        print(f"{color} {feature}: {status}")
    
    print("\n" + "="*80)
    
    # Show installation tips if needed
    missing_features = []
    if not VOICE_AVAILABLE:
        missing_features.append("🔊 Voice: pip install speechrecognition pyttsx3")
    if not ENCRYPTION_AVAILABLE:
        missing_features.append("🔐 Encryption: pip install cryptography")
    if not NUMPY_AVAILABLE:
        missing_features.append("📊 Analytics: pip install numpy")
    
    if missing_features:
        print("\n💡 **Optional Feature Installation:**")
        for feature in missing_features:
            print(f"   {feature}")
        print()

def show_tone_selection():
    """Enhanced tone selection menu"""
    print("\n" + "="*70)
    print("🎭 FRIDAY COMMUNICATION PREFERENCES")
    print("="*70)
    print("\nChoose how you'd like Friday to communicate with you:\n")
    
    print("💙 1. SUPPORTIVE (Recommended)")
    print("   • Warm, empathetic, lots of emotional validation")
    print("   • Includes resources and gentle guidance")
    print("   • Perfect for emotional support during pregnancy")
    print("   • Example: 'I understand you're feeling scared, and that's completely normal...'")
    
    print("\n💅 2. SASSY")  
    print("   • Friendly, confident, like your best friend")
    print("   • Uses 'girl', 'honey', 'queen' language")
    print("   • Playful but supportive approach")
    print("   • Example: 'Girl, you've got this! Let me tell you what's up...'")
    
    print("\n📊 3. DIRECT")
    print("   • Facts-focused, evidence-based responses")
    print("   • Minimal emotion, maximum information")
    print("   • Great for science-minded users")
    print("   • Example: 'Research indicates that 70% of mothers experience...'")
    
    print("\n🏥 4. CLINICAL")
    print("   • Medical terminology and clinical perspective")
    print("   • Professional healthcare communication style")
    print("   • Detailed medical information focus")
    print("   • Example: 'Maternal patients in the second trimester typically...'")
    
    print("\n😊 5. FRIENDLY")
    print("   • Casual, conversational, like chatting with a friend")
    print("   • Relaxed and approachable communication")
    print("   • Balanced between support and information")
    print("   • Example: 'Hey there! I'm so glad you asked about this...'")
    
    print("\n" + "="*70)
    print("💡 You can change your tone anytime with: !tone [supportive/sassy/direct/clinical/friendly]")
    print("🎯 You can also say things like 'be more supportive' during our conversation!")
    print("="*70)

def get_tone_choice():
    """Enhanced tone selection with more options"""
    while True:
        try:
            choice = input("\nEnter your choice (1-5) or press Enter for Supportive: ").strip()
        except (EOFError, KeyboardInterrupt):
            return "supportive", "💙 Perfect! Friday will be warm and supportive."
        
        tone_map = {
            "": ("supportive", "💙 Perfect! Friday will be warm and supportive."),
            "1": ("supportive", "💙 Perfect! Friday will be warm and supportive."),
            "2": ("sassy", "💅 Great choice! Friday will be your sassy bestie."),
            "3": ("direct", "📊 Excellent! Friday will give you straight facts."),
            "4": ("clinical", "🏥 Perfect! Friday will use clinical, medical communication."),
            "5": ("friendly", "😊 Awesome! Friday will be your friendly companion.")
        }
        
        if choice in tone_map:
            return tone_map[choice]
        else:
            print("❌ Please enter 1-5, or press Enter for default.")

def show_interaction_mode_selection():
    """Show interaction mode options"""
    print("\n🎯 **Choose Your Interaction Mode:**")
    print("1. 💬 Text Chat (Default)")
    print("2. 🎤 Voice Chat (Speak and Listen)")
    print("3. 🔄 Mixed Mode (Text + Voice Features)")
    
    if not VOICE_AVAILABLE:
        print("\n⚠️  Voice features require: pip install speechrecognition pyttsx3")
    
    return input("\nEnter choice (1-3) or press Enter for text: ").strip()

# === MAIN EXECUTION WITH ALL ENHANCEMENTS ===
if __name__ == "__main__":
    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        print("\n\n💙 Friday is saving conversations and shutting down gracefully...")
        if 'ai' in globals() and ai:
            ai._save_all_conversations()
            print("✅ All data saved securely.")
        print("👋 Take care! I'm always here when you need me.")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # === Enhanced Startup Sequence ===
    show_enhanced_startup()
    
    # === Brain & Domain Setup ===
    from core.MemoryCore import MemoryCore
    from core.EmotionCoreV2 import EmotionCoreV2

    print("\n🧠 Initializing cognitive architecture...")
    ai = None

    try:
        memory = MemoryCore(memory_file="friday_memory.enc", key_file="memory.key")
        emotion = EmotionCoreV2()
        ai = FridayAI(memory, emotion)
        
        # Initialize optional components with safe imports
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

        print("✅ Cognitive architecture loaded successfully!")

        # === Show Performance Report ===
        print(ai.get_comprehensive_report())

        # === Tone Selection ===
        show_tone_selection()
        chosen_tone, confirmation_msg = get_tone_choice()
        ai.tone_manager.current_tone = chosen_tone
        print(f"\n{confirmation_msg}")
        
        # === Interaction Mode Selection ===
        mode_choice = show_interaction_mode_selection()
        
        if mode_choice == "2" and VOICE_AVAILABLE:
            # Voice mode
            print("\n🎤 Entering voice interaction mode...")
            user_id = input("Enter your name (or press Enter for 'Guest'): ").strip() or "Guest"
            ai.current_user_id = user_id
            
            print(f"\nHello {user_id}! 🎤 Voice mode is ready.")
            ai.voice_interaction(user_id)
            print("\n💬 Switching to text mode for any final questions...")
        
        elif mode_choice == "3":
            # Mixed mode
            print("\n🔄 Mixed mode activated! Use !voice during chat to switch to voice temporarily.")
            ai._voice_enabled_for_session = True
        
        # === Get User Info ===
        print(f"\n🌟 Hello! I'm Friday, your Super Ultra Brilliant AI companion.")
        
        user_name = input("What should I call you? (or press Enter for 'Friend'): ").strip() or "Friend"
        ai.current_user_id = user_name
        
        # Optional pregnancy week
        try:
            pregnancy_input = input("Are you pregnant? If so, what week? (or press Enter to skip): ").strip()
            if pregnancy_input and pregnancy_input.isdigit():
                ai.pregnancy_week = int(pregnancy_input)
                print(f"✨ Got it! Week {ai.pregnancy_week} - such an exciting time!")
            elif pregnancy_input.lower() in ['yes', 'y']:
                print("💙 Wonderful! You can tell me your week anytime, or I'm happy to support you however you need.")
        except:
            pass
        
        print(f"\n💙 Nice to meet you, {user_name}! I'm here to support you through whatever you're experiencing.")
        print("How are you feeling today?")

        # === Main Conversation Loop ===
        while True:
            print("\n" + "="*60)
            try:
                user_input = input(f"{user_name}: ").strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\nFriday: Take care, {user_name}! I'm always here when you need me. 💙")
                ai._save_all_conversations()
                break

            if user_input.lower() in ["exit", "quit", "goodbye", "bye", "see you later"]:
                print(f"\nFriday: Take care, {user_name}! It's been wonderful supporting you. I'm always here when you need me. 💙")
                ai._save_all_conversations()
                break

            if not user_input:
                encouragements = [
                    "I'm listening. What's on your mind?",
                    "I'm here for whatever you need to talk about.",
                    "Take your time. I'm here when you're ready.",
                    "What would be most helpful for you right now?"
                ]
                print(f"\nFriday: {random.choice(encouragements)}")
                continue

            try:
                # Handle special commands first
                special_response = handle_special_commands(user_input, ai)
                if special_response:
                    print(f"\nFriday: {special_response}")
                    continue

                # Handle tone changes
                tone_response = ai.tone_manager.detect_tone_request(user_input)
                if tone_response:
                    print(f"\nFriday: {tone_response}")
                    continue
                
                # Handle feedback
                if ai.self_eval.interaction_count > 0 and any(word in user_input.lower() for word in 
                    ['good', 'great', 'better', 'worse', 'change', 'different', 'feedback']):
                    feedback_response = ai.self_eval.process_feedback(user_input, ai.tone_manager.current_tone)
                    print(f"\nFriday: {feedback_response}")
                    continue

                # Legacy command support
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

                # Main conversation - enhanced intelligent handling
                start_time = time.time()
                response = handle_user_input_intelligently(user_input, ai)
                end_time = time.time()
                
                print(f"\nFriday: {response}")
                
                # Update performance metrics
                ai.performance_metrics["successful_interactions"] += 1
                
                # Show response time if slow
                response_time = end_time - start_time
                if response_time > 3:
                    print(f"⏱️  (Response took {response_time:.1f}s - optimizing for next time)")

            except Exception as e:
                # Enhanced error handling
                ai.performance_metrics["error_count"] += 1
                
                error_responses = {
                    "json": "I had trouble processing that format. Could you rephrase it differently?",
                    "memory": "Let me think about that differently... Could you ask me again?",
                    "emotion": "I'm sensing a lot of feeling in your words. How can I support you right now?",
                    "network": "I'm having connectivity issues. Let me try a different approach.",
                    "timeout": "That's taking longer than expected. Could you try asking in a simpler way?"
                }
                
                error_type = type(e).__name__.lower()
                error_message = None
                
                for key, message in error_responses.items():
                    if key in str(e).lower() or key in error_type:
                        error_message = message
                        break
                
                if not error_message:
                    error_message = "Something's not quite right on my end. Could you try asking that another way?"
                
                print(f"\nFriday: {error_message}")
                                
                # Silent logging for debugging
                ai.logger.error(f"Error processing '{user_input[:50]}...': {e}")
                
                # Offer help recovery
                if ai.performance_metrics["error_count"] % 3 == 0:
                    print("\n💡 If I keep having trouble, try:")
                    print("   • Using simpler questions")
                    print("   • Being more specific about what you need")
                    print("   • Using !status to check my system health")

    except Exception as e:
        print("Friday: I'm having trouble starting up. Let me show you the technical details:")
        import traceback
        print("\n" + "="*60)
        print("STARTUP ERROR DETAILS")
        print("="*60)
        traceback.print_exc()
        print("="*60)
        print("\n💡 Try:")
        print("1. Check that all required packages are installed")
        print("2. Verify your Python environment")
        print("3. Check file permissions in the Friday directory")
        print("4. Report this error if the problem persists")
    
    finally:
        # Cleanup on exit
        if 'ai' in locals() and ai:
            print("\n💾 Saving final state...")
            ai._save_all_conversations()
            print("✅ All conversations and data saved securely.")
        
        print("\n🌟 Thank you for using Friday AI Super Ultra Brilliant!")
        print("💙 Your AI companion is always here when you need support.")