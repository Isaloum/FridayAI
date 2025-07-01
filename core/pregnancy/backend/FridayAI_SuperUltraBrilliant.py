# =====================================
# File: FridayAI_SuperUltraBrilliant.py (ULTIMATE MERGE - STEP 2 UPDATED)
# Purpose: Complete merge with modularized Unstoppable Features
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
    
PREGNANCY_EMOTION_AVAILABLE = True  # Will be handled by pregnancy_intelligence module

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
from pregnancy_intelligence import PregnancyIntelligence
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

# ====== MODULAR IMPORTS ======
from legendary_features import LegendaryConversationMemory, GoalCoachingSystem
from unstoppable_features import UnstoppableFeatures, ConversationState

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
        self.agents = {}
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
      
        
        # === LEGENDARY ENHANCEMENTS ===
        self.legendary_memory = LegendaryConversationMemory()
        self.goal_coach = GoalCoachingSystem()
        self.output_formatter = RichOutputFormatter()
        self.self_eval = SelfEvaluationSystem()
        self.citation_system = CitationSystem()
        self.knowledge_injection = KnowledgeInjectionSystem()
        
        # === UNSTOPPABLE ENHANCEMENTS (MODULARIZED) ===
        self.unstoppable = UnstoppableFeatures()
        
        # Session tracking
        self.session_topics = []
        self.current_user_id = "default"
        self.pregnancy_week = 0
        
        self._configure_logging()
        self._init_components()
        self._init_knowledge_systems()
        self.identity = SelfNarrativeCore()
        self.pregnancy = PregnancyIntelligence(self.memory, self.emotion, self.identity)
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
    
    def add_agent(self, agent):
        """Register a specialized agent"""
        self.agents[agent.__class__.__name__] = agent
        
    def handle_api_request(self, request_data):
        # Let agents handle specialized requests first
        for agent in self.agents.values():
            if agent.should_handle(request_data.get('message', '')):
                return agent.process(request_data)
        
        # Default processing
        analysis = self.analyze_input_semantic(request_data['message'])
        response = self.generate_response(
            request_data['message'],
            analysis,
            request_data.get('user_name', ''),
            request_data.get('pregnancy_week', 0),
            request_data.get('tone', 'supportive')
        )
        
        # Add agent-specific enhancements
        if 'PregnancyEmotionPlanner' in self.agents:
            response['content'] = self.agents['PregnancyEmotionPlanner'].enhance_response(
                response['content'],
                analysis.get('emotions', [])
            )
            
        return response

    
    def generate_response(self, user_input):
        """Main response generation method"""
        # Memory injection
        ctx = inject(user_input)
            
        # Knowledge citations
        citations = query_knowledge(user_input)
        excluded_files = ['requirements.txt', 'cognition_notes.txt', '.gitignore', '.env']
            
        # Generate response
        result = self.pipeline.generate_response(user_input)
            
        # Handle response format
        return result       
        
    def _analyze_input_semantic(self, user_input):
        """Enhanced semantic analysis with emergency detection"""
        input_lower = user_input.lower().strip()
        
        # 1. EMERGENCY CHECK FIRST (using unstoppable features)
        is_emergency, emergency_type, urgency_level = self.unstoppable.analyze_input_for_emergencies(user_input)
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
            bool(re.match(r'^[a-zA-Z]{1,4}$', input_lower)),
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
                    self.unstoppable.cleanup_old_data()  # Use unstoppable features cleanup
                except Exception as e:
                    self.logger.error(f"Cleanup error: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup, daemon=True)
        cleanup_thread.start()
    
    def _save_all_conversations(self):
        """Save all active conversations and state"""
        # Save legendary memory
        if hasattr(self, 'legendary_memory'):
            try:
                memory_path = Path("conversations/legendary_memory.pkl")
                memory_path.parent.mkdir(exist_ok=True)
                with open(memory_path, 'wb') as f:
                    pickle.dump(self.legendary_memory, f)
            except:
                pass
    
    def _save_performance_metrics(self):
        """Save performance metrics"""
        try:
            metrics_path = Path("performance_metrics.json")
            metrics_data = {
                'uptime_start': self.unstoppable.performance_metrics['uptime_start'].isoformat(),
                'successful_interactions': self.unstoppable.performance_metrics['successful_interactions'],
                'error_count': self.unstoppable.performance_metrics['error_count'],
                'emergency_responses': self.unstoppable.performance_metrics['emergency_responses'],
                'vault_operations': self.unstoppable.performance_metrics['vault_operations'],
                'predictions_made': self.unstoppable.performance_metrics['predictions_made']
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
            self.unstoppable.performance_metrics['vault_operations'] += 1
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
                'voice_capable': VOICE_AVAILABLE,
                'secure_vault': ENCRYPTION_AVAILABLE
            },
            'unstoppable_features': {
                'resilience_active': True,
                'predictive_analytics': True,
                'emergency_protocol': True,
                'secure_vault': self.unstoppable.vault.enabled
            },
            'session_data': {
                'user_id': self.current_user_id,
                'interaction_count': self.unstoppable.performance_metrics['successful_interactions'],
                'session_topics': self.session_topics[-5:] if self.session_topics else [],
                'current_tone': self.tone_manager.current_tone
            },
            'additional_data': additional_data or {}
        }

    # === MAIN RESPOND_TO METHOD WITH ALL ENHANCEMENTS ===
    @property
    def voice(self):
        """Voice interface property for backward compatibility"""
        if not hasattr(self, '_voice'):
            self._voice = VoiceInterface()
        return self._voice

    def respond_to(self, user_input: str, pregnancy_week: int = 0) -> Dict[str, object]:
        """SUPER ULTRA BRILLIANT respond_to with ALL enhancement features integrated"""
        
        start_time = time.time()
        
        # Store pregnancy week
        if pregnancy_week > 0:
            self.pregnancy_week = pregnancy_week
        
        # === EMERGENCY CHECK FIRST (using unstoppable features) ===
        analysis = self._analyze_input_semantic(user_input)
        
        if analysis.get('type') == 'emergency':
            emergency_response = self.unstoppable.generate_emergency_response(
                analysis['emergency_type'], 
                analysis['urgency_level']
            )
            
            # Voice alert for emergencies
            if self.voice.enabled:
                self.voice.speak("This appears to be an emergency. Please seek immediate medical attention.", interrupt_current=True)
            
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
        @self.unstoppable.resilience.wrap_with_resilience
        def generate_core_response():
            # Memory injection
            ctx = inject(user_input)
            
            # Knowledge citations
            citations = query_knowledge(user_input)
            excluded_files = ['requirements.txt', 'cognition_notes.txt', '.gitignore', '.env']
            
            
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
        
        # 2. PREDICTIVE MILESTONE INTEGRATION (using unstoppable features)
        if self.pregnancy_week > 0:
            milestones = self.unstoppable.get_milestone_predictions(self.pregnancy_week, 3)
            if milestones and ('milestone' in user_input.lower() or random.random() < 0.15):
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
            self.unstoppable.performance_metrics['predictions_made'] += 1
        
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
        
        # Performance metrics (using unstoppable features)
        end_time = time.time()
        response_time = end_time - start_time
        self.unstoppable.performance_metrics["response_times"].append(response_time)
        self.unstoppable.performance_metrics["successful_interactions"] += 1
        
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
        """Enhanced health insights from stored data (using unstoppable features)"""
        return self.unstoppable.get_user_health_summary(user_id)
    
    def get_comprehensive_report(self) -> str:
        """Generate comprehensive system performance report"""
        
        # Get base system report from unstoppable features
        unstoppable_report = self.unstoppable.get_system_health_report()
        
        # Add legendary features status
        legendary_status = f"""
🏆 **Legendary Features Status:**
• 💭 Conversation Memory: ✅ Active ({len(self.legendary_memory.conversations)} conversations)
• 🎯 Goal Coaching: ✅ Active ({len(self.goal_coach.active_goals.get(self.current_user_id, []))} active goals)
• 📝 Self-Evaluation: ✅ Active ({self.self_eval.interaction_count} interactions tracked)
• 📚 Citation System: ✅ Active ({sum(self.citation_system.usage_tracking.values())} citations provided)
• 💡 Knowledge Injection: ✅ Active ({sum(self.knowledge_injection.fact_preferences.values())} facts shared)
• 🎨 Rich Formatting: ✅ Active
• 🎭 Tone Management: ✅ Active (Current: {self.tone_manager.current_tone.title()})
"""
        
        # Combine reports
        combined_report = f"""
🏆 **Friday AI Super Ultra Brilliant - Complete System Report**

{legendary_status}

{unstoppable_report}

**Integration Status:**
• 🧠 Modular Architecture: ✅ Phase 1, Step 2 Complete
• 🔗 Feature Integration: ✅ All modules connected
• 🔄 Cross-Module Communication: ✅ Operational
• 📊 Unified Performance Tracking: ✅ Active

**Next Development Phase:**
• 🎯 Phase 1, Step 3: Extract Pregnancy Intelligence
• 🧠 Phase 1, Step 4: Extract Core Cognitive
• 🏆 Phase 1, Step 5: Create Clean Main Brain

**Status:** 🟢 All systems operational and modularized
**Version:** Super Ultra Brilliant v1.0 (Modular)
"""
        
        return combined_report
    
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
            },
            'legendary_features': {
                'conversation_count': len(self.legendary_memory.conversations),
                'emotional_insights': self.legendary_memory.get_emotional_insights(),
                'self_evaluation_data': {
                    'interaction_count': self.self_eval.interaction_count,
                    'feedback_count': len(self.self_eval.user_feedback_history)
                }
            }
        }
        
        # Add conversation history
        for conv in self.legendary_memory.conversations:
            export_data['conversation_history'].append({
                'timestamp': conv['timestamp'].isoformat(),
                'user_input': conv['user_input'],
                'ai_response': conv['ai_response'],
                'emotional_tone': conv['emotional_tone']
            })
        
        # Add unstoppable features data
        unstoppable_data = self.unstoppable.export_all_user_data(user_id, include_health)
        export_data['unstoppable_features'] = unstoppable_data.get('unstoppable_features', {})
        
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
            if ai.unstoppable.store_user_health_data(ai.current_user_id, "manual_entry", data):
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
    
    elif user_input.lower() == "!unstoppable":
        return ai.unstoppable.get_system_health_report()
    
    elif user_input.lower() == "!emergency_checklist":
        return ai.unstoppable.get_emergency_checklist()
    
    return None

def show_enhanced_startup():
    """Show enhanced startup sequence"""
    print("\n" + "="*80)
    print("🏆 FRIDAY AI - SUPER ULTRA BRILLIANT EDITION (MODULAR)")
    print("="*80)
    print("\n🚀 **Initializing Advanced Features:**")
    
    features = [
        ("🧠 Legendary Memory System", "LOADED"),
        ("🎯 Goal Coaching Engine", "ACTIVE"),
        ("🛡️ Unstoppable Features", "MODULARIZED"),
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
        color = "🟢" if status == "ACTIVE" or status == "LOADED" or status == "AVAILABLE" or status == "MODULARIZED" else "🟡" if "INSTALL" in status else "🟢"
        print(f"{color} {feature}: {status}")
    
    print("\n" + "="*80)
    print("📈 **Modularization Progress:**")
    print("✅ Step 1: Legendary Features → legendary_features.py")
    print("✅ Step 2: Unstoppable Features → unstoppable_features.py")
    print("⏳ Step 3: Pregnancy Intelligence → (Next)")
    print("⏳ Step 4: Core Cognitive → (Next)")
    print("⏳ Step 5: Clean Main Brain → (Next)")
    print("="*80)
    
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
                # Main conversation only - no keyword triggers
                start_time = time.time()
                response = ai.generate_response(user_input)
                end_time = time.time()
                
                print(f"\nFriday: {response}")
                
                # Update performance metrics
                ai.unstoppable.performance_metrics["successful_interactions"] += 1
                
                # Show response time if slow
                response_time = end_time - start_time
                if response_time > 3:
                    print(f"⏱️  (Response took {response_time:.1f}s - optimizing for next time)")

                # Dynamic name detection using AI
                name_response = ai.pipeline.llm.prompt(f"Does this message contain a name the person wants to be called? Just answer the name or 'none': '{user_input}'")

                if isinstance(name_response, dict):
                    detected_name = name_response.get('reply', '').strip()
                else:
                    detected_name = str(name_response).strip()

                # Update name if AI detected one
                if detected_name.lower() not in ['none', 'no', '']:
                    clean_name = detected_name.split()[0].strip(".,!?\"'")
                    if clean_name and len(clean_name) < 20:
                        user_name = clean_name.capitalize()
                        print(f"[Name updated to: {user_name}]")
                                    
            except Exception as e:
                # Enhanced error handling using unstoppable features
                ai.unstoppable.performance_metrics["error_count"] += 1
                
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
                if ai.unstoppable.performance_metrics["error_count"] % 3 == 0:
                    print("\n💡 If I keep having trouble, try:")
                    print("   • Using simpler questions")
                    print("   • Being more specific about what you need")
                    print("   • Using !status to check my system health")
                    print("   • Using !unstoppable to check resilience systems")
            
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
        print("🏆 Phase 1, Step 2 Complete: Unstoppable Features Successfully Modularized!")
        
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Initialize the Flask App ---
app = Flask(__name__)
CORS(app) # This allows your React app to talk to this server

# --- Initialize Friday AI Brain ---
# This assumes your main AI class and its setup are available
# You already have this code, so just ensure it's loaded before this point
memory = MemoryCore(memory_file="friday_memory.enc", key_file="memory.key")
emotion = EmotionCoreV2()
friday_ai_instance = FridayAI(memory, emotion)


# --- Define the API Endpoint ---
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message")

    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    # Use your Friday AI brain to get a real response
    ai_response = friday_ai_instance.respond_to(user_input)

    # Send the real response back to the React front-end
    return jsonify(ai_response)


# --- Run the Server ---
if __name__ == "__main__":
    # Make sure to remove your old command-line input loop
    print("Starting Friday AI Flask server on http://localhost:5000")
    app.run(port=5000, debug=False)