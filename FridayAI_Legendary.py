# =====================================
# File: FridayAI_Legendary.py (Complete Integration) - YOUR ORIGINAL + LEGENDARY FEATURES
# Purpose: Contains the complete cognitive architecture of Friday + ALL legendary enhancements
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
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

# === LEGENDARY ENHANCEMENT IMPORTS ===
import random
import json
import hashlib
from dataclasses import dataclass
from collections import defaultdict, deque

# === External Core Modules (ALL YOUR ORIGINAL IMPORTS) ===
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

# === Legacy/Non-core Modules (ALL YOUR ORIGINAL IMPORTS) ===
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

# === BOOTSTRAP ===
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

# === LEGENDARY ENHANCEMENT CLASSES ===

@dataclass
class LegendaryConversationMemory:
    """Enhanced conversation memory with context awareness"""
    def __init__(self, max_history=50):
        self.conversations = deque(maxlen=max_history)
        self.topic_memory = defaultdict(list)  # topic -> list of (question, answer, timestamp)
        self.user_patterns = defaultdict(int)  # track user conversation patterns
    
    def add_exchange(self, user_input: str, ai_response: str, emotional_tone: str):
        """Store conversation exchange with enhanced metadata"""
        timestamp = datetime.now()
        
        # Create conversation entry
        entry = {
            'user_input': user_input,
            'ai_response': ai_response,
            'emotional_tone': emotional_tone,
            'timestamp': timestamp,
            'topic_hash': self._get_topic_hash(user_input)
        }
        
        self.conversations.append(entry)
        
        # Extract and store topic
        topic = self._extract_topic(user_input)
        if topic:
            self.topic_memory[topic].append(entry)
        
        # Track user patterns
        self.user_patterns[emotional_tone] += 1
    
    def find_similar_conversation(self, user_input: str, similarity_threshold=0.7) -> Optional[Dict]:
        """Find similar previous conversations using semantic matching"""
        current_topic_hash = self._get_topic_hash(user_input)
        current_keywords = set(self._extract_keywords(user_input))
        
        best_match = None
        best_score = 0
        
        for conv in reversed(list(self.conversations)):  # Start with most recent
            # Skip very recent conversations (within last 2 exchanges)
            if len(self.conversations) > 2 and conv == list(self.conversations)[-2]:
                continue
                
            past_keywords = set(self._extract_keywords(conv['user_input']))
            
            # Calculate similarity score
            if len(current_keywords) > 0 and len(past_keywords) > 0:
                intersection = current_keywords.intersection(past_keywords)
                union = current_keywords.union(past_keywords)
                jaccard_score = len(intersection) / len(union)
                
                # Boost score for emotional similarity
                if conv['emotional_tone'] in user_input.lower():
                    jaccard_score += 0.2
                
                if jaccard_score > similarity_threshold and jaccard_score > best_score:
                    best_match = conv
                    best_score = jaccard_score
        
        return best_match if best_match else None
    
    def _get_topic_hash(self, text: str) -> str:
        """Generate topic hash for grouping similar conversations"""
        keywords = self._extract_keywords(text)
        topic_string = ' '.join(sorted(keywords[:3]))  # Top 3 keywords
        return hashlib.md5(topic_string.encode()).hexdigest()[:8]
    
    def _extract_topic(self, text: str) -> Optional[str]:
        """Extract main topic from user input"""
        pregnancy_topics = ['pregnancy', 'baby', 'birth', 'labor', 'prenatal', 'trimester']
        health_topics = ['anxiety', 'depression', 'stress', 'mood', 'sleep']
        
        text_lower = text.lower()
        for topic in pregnancy_topics + health_topics:
            if topic in text_lower:
                return topic
        return None
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        # Remove common stop words and extract meaningful terms
        stop_words = {'i', 'am', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = re.findall(r'\b\w+\b', text.lower())
        return [word for word in words if len(word) > 2 and word not in stop_words]

class GoalCoachingSystem:
    """Multi-turn goal and task planning system"""
    def __init__(self):
        self.active_goals = {}  # user_id -> list of goals
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
        self.pending_check_ins = {}  # goal_id -> next_check_date
    
    def detect_goal_opportunity(self, user_input: str, ai_response: str) -> Optional[str]:
        """Detect if user might benefit from goal setting"""
        goal_triggers = {
            'anxiety_management': ['anxious', 'worried', 'scared', 'stress', 'overwhelmed'],
            'birth_preparation': ['birth', 'delivery', 'labor', 'due date', 'hospital'],
            'nutrition_health': ['eating', 'nutrition', 'weight', 'vitamins', 'healthy']
        }
        
        user_lower = user_input.lower()
        
        # Check if response includes advice/resources (good time to offer coaching)
        if any(word in ai_response.lower() for word in ['try', 'consider', 'help', 'suggest', 'recommend']):
            for goal_type, triggers in goal_triggers.items():
                if any(trigger in user_lower for trigger in triggers):
                    return goal_type
        
        return None
    
    def create_goal_offer(self, goal_type: str) -> str:
        """Create a goal coaching offer"""
        template = self.goal_templates.get(goal_type)
        if not template:
            return ""
        
        offer = f"\n\n🎯 **Would you like me to help you create a personal plan?**\n"
        offer += f"I could help you work on: **{template['title']}**\n\n"
        offer += "This would include:\n"
        for i, step in enumerate(template['steps'][:3], 1):  # Show first 3 steps
            offer += f"• {step}\n"
        offer += f"\nI'd check in with you every {template['check_in_days']} days to see how you're doing.\n"
        offer += "**Interested? Just say 'yes' or 'create goal'!**"
        
        return offer
    
    def create_goal(self, goal_type: str, user_id: str = "default") -> str:
        """Create and activate a new goal"""
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
        
        # Schedule first check-in
        next_check = datetime.now() + timedelta(days=template['check_in_days'])
        self.pending_check_ins[goal_id] = next_check
        
        response = f"🎯 **Goal Created: {template['title']}**\n\n"
        response += "Your action steps:\n"
        for i, step in enumerate(template['steps'], 1):
            response += f"{i}. {step}\n"
        response += f"\n💙 I'll check in with you in {template['check_in_days']} days to see how you're progressing!"
        
        return response
    
    def check_for_due_check_ins(self, user_id: str = "default") -> Optional[str]:
        """Check if any goals are due for check-in"""
        now = datetime.now()
        due_check_ins = []
        
        for goal_id, check_date in self.pending_check_ins.items():
            if now >= check_date:
                # Find the goal
                for goal in self.active_goals.get(user_id, []):
                    if goal['id'] == goal_id:
                        due_check_ins.append(goal)
                        break
        
        if due_check_ins:
            goal = due_check_ins[0]  # Handle one at a time
            return self._create_check_in_message(goal)
        
        return None
    
    def _create_check_in_message(self, goal: Dict) -> str:
        """Create a check-in message for a goal"""
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
    """Enhanced output formatting with ANSI colors and structure"""
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
        """Check if terminal supports ANSI colors"""
        return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty() and os.name != 'nt'
    
    def format_response(self, text: str, response_type: str = 'normal') -> str:
        """Format response with appropriate styling"""
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
        """Format empathy responses with warm colors"""
        # Highlight emotional validation phrases
        text = re.sub(r'(I understand|I hear you|That sounds|It\'s normal)', 
                     f"{self.colors['empathy']}\\1{self.colors['end']}", text)
        
        # Make headers bold
        text = re.sub(r'(\*\*.*?\*\*)', 
                     f"{self.colors['bold']}\\1{self.colors['end']}", text)
        
        return text
    
    def _format_goal(self, text: str) -> str:
        """Format goal-related content"""
        # Highlight goal titles
        text = re.sub(r'🎯 \*\*(.*?)\*\*', 
                     f"🎯 {self.colors['bold']}{self.colors['success']}\\1{self.colors['end']}", text)
        
        # Color check marks
        text = text.replace('✅', f"{self.colors['success']}✅{self.colors['end']}")
        
        return text
    
    def _format_resource(self, text: str) -> str:
        """Format resource recommendations"""
        # Highlight resource categories
        text = re.sub(r'(\*\*📚 .*?\*\*|\*\*📱 .*?\*\*|\*\*🤝 .*?\*\*)', 
                     f"{self.colors['info']}\\1{self.colors['end']}", text)
        
        return text
    
    def _format_normal(self, text: str) -> str:
        """Format normal responses"""
        # Just make bold text actually bold
        text = re.sub(r'\*\*(.*?)\*\*', 
                     f"{self.colors['bold']}\\1{self.colors['end']}", text)
        
        return text
    
    def _format_plain(self, text: str) -> str:
        """Format for terminals that don't support colors"""
        # Convert markdown-style bold to plain text emphasis
        text = re.sub(r'\*\*(.*?)\*\*', r'[\1]', text)
        return text

class SelfEvaluationSystem:
    """Self-evaluation and feedback system"""
    def __init__(self):
        self.interaction_count = 0
        self.feedback_requests = 0
        self.user_feedback_history = []
        self.tone_adjustments = defaultdict(int)
        self.last_feedback_request = None
    
    def should_request_feedback(self) -> bool:
        """Determine if Friday should ask for feedback"""
        self.interaction_count += 1
        
        # Don't ask too frequently
        if self.last_feedback_request:
            time_since_last = datetime.now() - self.last_feedback_request
            if time_since_last.total_seconds() < 600:  # 10 minutes minimum
                return False
        
        # Ask after certain interaction milestones
        if self.interaction_count in [7, 20, 50] or (self.interaction_count > 50 and self.interaction_count % 25 == 0):
            return True
        
        # Random chance after 10 interactions
        if self.interaction_count > 10 and random.random() < 0.08:  # 8% chance
            return True
        
        return False
    
    def generate_feedback_request(self, current_tone: str, recent_topics: List[str]) -> str:
        """Generate a context-aware feedback request"""
        self.feedback_requests += 1
        self.last_feedback_request = datetime.now()
        
        # Vary the feedback request based on context
        requests = [
            f"💭 Quick check-in: How am I doing with my {current_tone} tone? Should I adjust anything?",
            f"🎯 I want to make sure I'm helping you well. How's our conversation style working for you?",
            f"💙 Am I being too {current_tone}, or would you prefer a different approach?",
            f"🔄 We've covered {', '.join(recent_topics[:2])} today. How can I better support you?"
        ]
        
        return random.choice(requests)
    
    def process_feedback(self, feedback: str, current_tone: str) -> str:
        """Process user feedback and adjust behavior"""
        feedback_lower = feedback.lower()
        
        self.user_feedback_history.append({
            'feedback': feedback,
            'timestamp': datetime.now(),
            'context_tone': current_tone
        })
        
        # Analyze feedback sentiment
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

class VoiceIntegrationSystem:
    """Optional voice integration for TTS"""
    def __init__(self):
        self.tts_enabled = False
        self.voice_settings = {
            'speed': 1.0,
            'voice': 'female',
            'pitch': 1.0
        }
    
    def toggle_voice(self, enable: bool = None) -> str:
        """Toggle voice output on/off"""
        if enable is None:
            self.tts_enabled = not self.tts_enabled
        else:
            self.tts_enabled = enable
        
        status = "enabled" if self.tts_enabled else "disabled"
        return f"🔊 Voice output {status}. {self._get_voice_help()}"
    
    def _get_voice_help(self) -> str:
        """Get voice command help"""
        if self.tts_enabled:
            return "I'll speak my responses aloud. Use !voice off to disable."
        else:
            return "Use !voice on to enable speech, or !voice settings to configure."
    
    def speak_response(self, text: str) -> bool:
        """Convert text to speech (placeholder - requires actual TTS library)"""
        if not self.tts_enabled:
            return False
        
        try:
            # Placeholder for actual TTS implementation
            # Would use libraries like gtts, pyttsx3, or OpenAI TTS
            # For now, just indicate that speech would happen
            print(f"🔊 [Speaking: {text[:50]}...]")
            return True
        except Exception as e:
            print(f"🔊 Voice error: {e}")
            return False

class CitationSystem:
    """Automatic citation and source linking"""
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
        """Add relevant citations to response"""
        response_lower = response.lower()
        
        # Find citation opportunities
        for pattern, citation_type in self.citation_patterns.items():
            if pattern in response_lower:
                citation = self._get_relevant_citation(topic_context, citation_type)
                if citation:
                    response += f"\n\n📚 **Source:** {citation}"
                    break  # Only add one citation per response
        
        return response
    
    def _get_relevant_citation(self, topic: str, citation_type: str) -> Optional[str]:
        """Get the most relevant citation for the topic"""
        topic_lower = topic.lower()
        
        # Match topic to sources
        for source_key, source_info in self.medical_sources.items():
            if any(keyword in topic_lower for keyword in source_info['domain'].split()):
                return f"[{source_info['title']}]({source_info['url']})"
        
        # Default citation for medical advice
        if citation_type in ['medical_advice', 'medical_research']:
            return "[American College of Obstetricians and Gynecologists](https://www.acog.org)"
        
        return None

class KnowledgeInjectionSystem:
    """Local knowledge injection with "Did You Know?" facts"""
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
        """Determine if a fact should be added"""
        # Don't add facts too frequently
        if self.last_fact_time:
            time_since = datetime.now() - self.last_fact_time
            if time_since.total_seconds() < 180:  # 3 minutes minimum
                return False
        
        # 15% chance of adding a fact
        return random.random() < 0.15
    
    def get_relevant_fact(self, topic_context: str, emotional_tone: str) -> str:
        """Get a relevant fact based on context"""
        self.last_fact_time = datetime.now()
        
        # Choose fact category based on context
        if any(word in topic_context.lower() for word in ['stress', 'anxiety', 'worried', 'overwhelmed']):
            facts_pool = self.wellness_facts
        else:
            facts_pool = self.pregnancy_facts
        
        # Get unused facts
        available_facts = [f for f in facts_pool if f not in self.used_facts]
        
        # Reset if all facts used
        if not available_facts:
            self.used_facts.clear()
            available_facts = facts_pool
        
        # Select fact
        fact = random.choice(available_facts)
        self.used_facts.add(fact)
        
        return f"\n\n💡 **Did you know?** {fact}"

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

# === FRIDAYAI CORE CLASS WITH LEGENDARY INTEGRATION ===
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
       
        # === Pregnancy Enhancement (PRESERVED) ===
        if PREGNANCY_EMOTION_AVAILABLE:
            self.pregnancy_emotion = PregnancyEmotionCore()
        else:
            self.pregnancy_emotion = None
        
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

        # === LEGENDARY ENHANCEMENTS INITIALIZATION ===
        self.legendary_memory = LegendaryConversationMemory()
        self.goal_coach = GoalCoachingSystem()
        self.formatter = RichOutputFormatter()
        self.self_eval = SelfEvaluationSystem()
        self.voice_system = VoiceIntegrationSystem()
        self.citation_system = CitationSystem()
        self.knowledge_injector = KnowledgeInjectionSystem()
        
        # Enhanced state tracking
        self.session_topics = []
        self.current_goals = []
        
        print("🏆 Friday AI with Legendary Features activated!")

    # === ALL YOUR ORIGINAL METHODS (PRESERVED) ===
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

    # === LEGENDARY HELPER METHODS ===
    def _handle_voice_command(self, command: str) -> str:
        """Handle voice-related commands"""
        parts = command.lower().split()
        if len(parts) == 1:
            return self.voice_system.toggle_voice()
        elif parts[1] == 'on':
            return self.voice_system.toggle_voice(True)
        elif parts[1] == 'off':
            return self.voice_system.toggle_voice(False)
        else:
            return "🔊 Voice commands: !voice [on/off]"

    def _handle_goal_creation(self) -> str:
        """Handle goal creation - can be customized based on recent conversation"""
        return self.goal_coach.create_goal('anxiety_management')

    def _determine_response_type(self, content: str, has_goal_offer: bool) -> str:
        """Determine response type for formatting"""
        if '💭 **I remember**' in content:
            return 'empathy'
        elif has_goal_offer or '🎯' in content:
            return 'goal'
        elif '📚' in content or '📖' in content:
            return 'resource'
        else:
            return 'normal'

    def _create_enhanced_response(self, content: str, response_type: str, user_input: str) -> Dict:
        """Create enhanced response format"""
        return {
            'domain': 'legendary_friday',
            'content': self.formatter.format_response(content, response_type),
            'confidence': 1.0,
            'emotional_tone': 'supportive',
            'processing_time': datetime.now().isoformat(),
            'legendary_features': {'active': True, 'type': response_type}
        }

    # === ENHANCED RESPOND_TO METHOD WITH LEGENDARY FEATURES ===
    def respond_to(self, user_input: str, pregnancy_week: int = 0) -> Dict[str, object]:
        """LEGENDARY respond_to with all enhancement features integrated"""
        
        # === LEGENDARY FEATURE CHECKS ===
        
        # Voice Integration Check
        if user_input.lower().startswith('!voice'):
            voice_response = self._handle_voice_command(user_input)
            return self._create_enhanced_response(voice_response, "system", user_input)
        
        # Goal Coaching Check
        if user_input.lower() in ['yes', 'create goal', 'set goal']:
            goal_response = self._handle_goal_creation()
            if goal_response:
                return self._create_enhanced_response(goal_response, "goal", user_input)
        
        # Check for due goal check-ins
        check_in = self.goal_coach.check_for_due_check_ins()
        if check_in:
            return self._create_enhanced_response(check_in, "goal", user_input)
        
        # Memory Recall Check (BEFORE processing)
        similar_conversation = self.legendary_memory.find_similar_conversation(user_input)
        
        # === YOUR ORIGINAL RESPONSE GENERATION (PRESERVED) ===
        
        # Check for tone change requests FIRST
        tone_response = self.tone_rewriter.detect_tone_request(user_input)
        if tone_response:
            return {
                "domain": "tone_selector",
                "content": tone_response,
                "confidence": 1.0,
                "emotional_tone": "neutral",
                "processing_time": datetime.now().isoformat()
            }
        
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
        
        # Enhance response with pregnancy awareness (silent)
        if pregnancy_analysis and pregnancy_analysis.intensity > 0.6:
            supportive_message = self.pregnancy_emotion.generate_supportive_response(pregnancy_analysis)
            raw_reply += f"\n\n{supportive_message}"
        
        # === LEGENDARY ENHANCEMENTS START HERE ===
        
        # 1. DYNAMIC MEMORY RECALL
        if similar_conversation:
            days_ago = (datetime.now() - similar_conversation['timestamp']).days
            time_ref = f"{days_ago} days ago" if days_ago > 0 else "earlier today"
            memory_prefix = f"💭 **I remember** we talked about something similar {time_ref}. "
            memory_prefix += f"You were feeling {similar_conversation['emotional_tone']} then too.\n\n"
            raw_reply = memory_prefix + raw_reply
        
        # 2. GOAL COACHING INTEGRATION
        goal_opportunity = self.goal_coach.detect_goal_opportunity(user_input, raw_reply)
        if goal_opportunity and not similar_conversation:  # Don't offer goals if referencing memory
            goal_offer = self.goal_coach.create_goal_offer(goal_opportunity)
            raw_reply += goal_offer
        
        # 3. ENHANCED RESOURCE OFFERING (keep your existing logic)
        resources_offer = self._offer_pregnancy_resources(user_input, emotional_tone)
        if resources_offer:
            raw_reply = resources_offer  # REPLACE instead of append
        
        # 4. CITATION SYSTEM
        topic_context = ' '.join(self.session_topics[-3:])  # Last 3 topics
        raw_reply = self.citation_system.add_citations(raw_reply, topic_context)
        
        # 5. KNOWLEDGE INJECTION
        if self.knowledge_injector.should_add_fact():
            fact = self.knowledge_injector.get_relevant_fact(user_input, emotional_tone)
            raw_reply += fact
        
        # 6. SELF-EVALUATION
        if self.self_eval.should_request_feedback():
            feedback_request = self.self_eval.generate_feedback_request(
                self.tone_manager.current_tone,
                self.session_topics[-2:]
            )
            raw_reply += f"\n\n{feedback_request}"
        
        # Apply tone rewriting to final output
        is_pregnancy_related = any(word in user_input.lower() for word in ["baby", "pregnant", "pregnancy", "mom", "mother"])
        if not resources_offer:  # Only apply tone if we're not showing empathy resources
            raw_reply = self.tone_rewriter.rewrite(raw_reply)
        
        # Apply tone if available  
        if hasattr(self, 'tone_manager'):
            raw_reply = self.tone_manager.apply_tone(raw_reply)
        
        # Only show relevant knowledge if it's truly helpful (filtered)
        relevant_citations = []
        for c in citations:
            if c.get('source') not in excluded_files and 'text' in c:
                # Only include if citation is actually relevant and substantial
                if len(c['text']) > 50 and any(word in c['text'].lower() for word in user_input.lower().split()):
                    relevant_citations.append(c)
        
        if relevant_citations and len(relevant_citations) <= 2:  # Limit to 2 most relevant
            sources = [f"📄 {c['text']}" for c in relevant_citations[:2]]
            raw_reply += "\n\n" + "\n\n".join(sources)
        
        # === LEGENDARY FORMATTING & VOICE ===
        response_type = self._determine_response_type(raw_reply, goal_opportunity)
        final_output = self.formatter.format_response(raw_reply, response_type)
        
        if self.voice_system.tts_enabled:
            self.voice_system.speak_response(raw_reply)
        
        # === STORE IN LEGENDARY MEMORY ===
        self.legendary_memory.add_exchange(user_input, raw_reply, emotional_tone)
        
        # Track session topics
        topic = self.legendary_memory._extract_topic(user_input)
        if topic:
            self.session_topics.append(topic)
        
        # === EXISTING LOGGING (PRESERVED) ===
        log_event(user_input, source="user")
        log_event(final_output, source="friday")
        
        try:
            update_mood(emotional_tone)
        except Exception as e:
            update_mood("neutral")
        
        # === LEGENDARY ENHANCED RESPONSE ===
        response = {
            "domain": "legendary_cognitive_pipeline",
            "content": final_output,
            "confidence": 1.0,
            "emotional_tone": emotional_tone,
            "memory_context": memory_context,
            "identity_context": identity_context,
            "processing_time": datetime.now().isoformat(),
            "current_tone": self.tone_rewriter.get_current_tone(),
            "legendary_features": {
                "memory_recall": similar_conversation is not None,
                "goal_coaching": goal_opportunity is not None,
                "rich_formatting": True,
                "citations_added": '📚' in raw_reply,
                "knowledge_injection": '💡' in raw_reply,
                "voice_enabled": self.voice_system.tts_enabled,
                "self_evaluation": self.self_eval.interaction_count,
                "session_topics": len(self.session_topics),
                "active_goals": len(self.goal_coach.active_goals.get("default", []))
            }
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

# === HELPER FUNCTIONS (PRESERVED WITH LEGENDARY ENHANCEMENTS) ===

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

# === MAIN EXECUTION WITH LEGENDARY FEATURES ===
if __name__ == "__main__":
    # === Brain & Domain Setup ===
    from core.MemoryCore import MemoryCore
    from core.EmotionCoreV2 import EmotionCoreV2

    # Silent initialization
    print("Friday is waking up with legendary capabilities...")
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

        # === TONE SELECTION AT STARTUP ===
        show_tone_selection()
        chosen_tone, confirmation_msg = get_tone_choice()
        
        # Set the chosen tone
        if hasattr(ai, 'tone_manager'):
            ai.tone_manager.current_tone = chosen_tone
        
        print(f"\n{confirmation_msg}")
        print(f"\n🏆 Hello! I'm Friday, your legendary AI companion with enhanced capabilities!")
        print("💭 I can remember our conversations, help you set goals, and adapt to your needs.")
        print("How are you feeling today?")

        while True:
            print("\n" + "="*50)
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nFriday: Take care! I'm always here when you need me. 💙")
                break

            if user_input.lower() in ["exit", "quit", "goodbye", "bye"]:
                print("\nFriday: Take care! I'm always here when you need me. 💙")
                break

            if not user_input:  # Empty input
                print("\nFriday: I'm listening. What's on your mind?")
                continue

            try:
                # === LEGENDARY COMMAND HANDLING ===
                if user_input.lower().startswith("!voice"):
                    response = ai._handle_voice_command(user_input)
                    print(f"\nFriday: {response}")
                    continue

                if user_input.lower().startswith("!goals"):
                    active_goals = len(ai.goal_coach.active_goals.get("default", []))
                    print(f"\nFriday: 🎯 You have {active_goals} active goals. Use 'check goals' to see details.")
                    continue

                if user_input.lower().startswith("!memory"):
                    conv_count = len(ai.legendary_memory.conversations)
                    print(f"\nFriday: 💭 I remember {conv_count} conversations from our time together.")
                    continue

                if user_input.lower().startswith("!stats"):
                    stats = f"""
🏆 **Legendary Friday Stats:**
💭 Conversations remembered: {len(ai.legendary_memory.conversations)}
🎯 Active goals: {len(ai.goal_coach.active_goals.get("default", []))}
📊 Feedback requests: {ai.self_eval.feedback_requests}
🔊 Voice enabled: {ai.voice_system.tts_enabled}
💡 Facts shared: {len(ai.knowledge_injector.used_facts)}
🎭 Current tone: {ai.tone_manager.current_tone}
📈 Interaction count: {ai.self_eval.interaction_count}
                    """
                    print(f"\nFriday: {stats}")
                    continue

                # Handle feedback responses
                if ai.self_eval.interaction_count > 0 and any(word in user_input.lower() for word in 
                    ['good', 'great', 'better', 'worse', 'change', 'different']):
                    feedback_response = ai.self_eval.process_feedback(user_input, ai.tone_manager.current_tone)
                    print(f"\nFriday: {feedback_response}")
                    continue

                # Check for tone changes FIRST
                if hasattr(ai, 'tone_manager'):
                    tone_response = ai.tone_manager.detect_tone_request(user_input)
                    if tone_response:
                        print(f"\nFriday: {tone_response}")
                        continue

                # Special commands
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

                # Main conversation - smart handling WITH LEGENDARY FEATURES
                response = handle_user_input_intelligently(user_input, ai)
                print(f"\nFriday: {response}")

            except Exception as e:
                # More graceful error handling
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