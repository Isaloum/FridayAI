# =====================================
# File: legendary_features.py
# Purpose: All Legendary Enhancement Classes extracted from main file
# =====================================

import random
import json
import hashlib
import time
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque

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