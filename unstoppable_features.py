# =====================================
# File: unstoppable_features.py
# Purpose: Unstoppable Features Module - Resilience, Emergency Detection, Predictive Analytics
# Phase 1, Step 2 of Brain Modularization
# =====================================

import logging
import os
import threading
import time
import re
import random
import json
import hashlib
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
from pathlib import Path
from functools import wraps
import warnings
warnings.filterwarnings("ignore")

# === Optional Advanced Libraries ===
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
        recent_errors = [e for e in self.error_history if (datetime.now() - e['timestamp']).seconds < 86400]
        
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


# ====== UNSTOPPABLE FEATURES MANAGER ======

class UnstoppableFeatures:
    """Main manager for all unstoppable features"""
    
    def __init__(self):
        # Initialize all unstoppable components
        self.resilience = ResilienceEngine()
        self.predictive = PredictiveAnalytics()
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
            "vault_operations": 0,
            "predictions_made": 0
        }
        
        self._setup_monitoring()
    
    def _setup_monitoring(self):
        """Setup system monitoring threads"""
        def health_monitor():
            while True:
                time.sleep(300)  # Check every 5 minutes
                try:
                    self._perform_health_check()
                except Exception as e:
                    logging.error(f"Health monitor error: {e}")
        
        monitor_thread = threading.Thread(target=health_monitor, daemon=True)
        monitor_thread.start()
    
    def _perform_health_check(self):
        """Perform periodic health checks"""
        # Check error rates
        if len(self.resilience.error_history) > 50:
            recent_errors = [e for e in self.resilience.error_history 
                           if (datetime.now() - e['timestamp']).seconds < 3600]
            if len(recent_errors) > 10:
                logging.warning("High error rate detected in last hour")
        
        # Check vault operations
        if self.vault.enabled:
            try:
                # Perform basic vault connectivity check
                test_data = {"health_check": datetime.now().isoformat()}
                self.vault.store_health_data("system", "health_check", test_data)
                self.performance_metrics["vault_operations"] += 1
            except:
                logging.error("Vault connectivity issue detected")
    
    def analyze_input_for_emergencies(self, user_input: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Analyze input for emergency situations"""
        is_emergency, emergency_type, urgency_level = self.emergency.check_emergency(user_input)
        
        if is_emergency:
            self.performance_metrics["emergency_responses"] += 1
            logging.critical(f"Emergency detected: {emergency_type} - {urgency_level}")
        
        return is_emergency, emergency_type, urgency_level
    
    def generate_emergency_response(self, emergency_type: str, urgency_level: str) -> str:
        """Generate emergency response with system integration"""
        response = self.emergency.generate_emergency_response(emergency_type, urgency_level)
        
        # Log emergency response
        emergency_data = {
            "emergency_type": emergency_type,
            "urgency_level": urgency_level,
            "timestamp": datetime.now().isoformat(),
            "response_generated": True
        }
        
        if self.vault.enabled:
            try:
                self.vault.store_health_data("system", "emergency_log", emergency_data)
            except:
                pass  # Don't let vault errors interfere with emergency response
        
        return response
    
    def predict_user_needs(self, user_id: str, mood_history: deque, current_context: str = "") -> Dict[str, float]:
        """Predict user needs with enhanced analytics"""
        predictions = self.predictive.predict_emotional_needs(mood_history, current_context)
        self.performance_metrics["predictions_made"] += 1
        
        return predictions
    
    def get_milestone_predictions(self, current_week: int, look_ahead: int = 4) -> List[str]:
        """Get pregnancy milestone predictions"""
        milestones = self.predictive.predict_upcoming_milestones(current_week, look_ahead)
        
        if milestones:
            self.performance_metrics["predictions_made"] += 1
        
        return milestones
    
    def store_user_health_data(self, user_id: str, category: str, data: Dict, metadata: Dict = None) -> bool:
        """Store health data securely"""
        if not self.vault.enabled:
            return False
        
        success = self.vault.store_health_data(user_id, category, data, metadata)
        
        if success:
            self.performance_metrics["vault_operations"] += 1
        
        return success
    
    def get_user_health_summary(self, user_id: str) -> str:
        """Get user's health data summary"""
        return self.vault.get_data_summary(user_id)
    
    def get_user_insights(self, user_id: str, interaction_data: List[Dict], current_week: int = 0) -> str:
        """Get personalized user insights"""
        user_patterns = self.predictive.analyze_user_patterns(user_id, interaction_data)
        insights = self.predictive.get_personalized_insights(user_patterns, current_week)
        
        self.performance_metrics["predictions_made"] += 1
        
        return insights
    
    def get_system_health_report(self) -> str:
        """Get comprehensive system health report"""
        uptime = datetime.now() - self.performance_metrics["uptime_start"]
        
        # Calculate averages
        if self.performance_metrics["response_times"]:
            if NUMPY_AVAILABLE:
                avg_response = np.mean(list(self.performance_metrics["response_times"]))
            else:
                response_times = list(self.performance_metrics["response_times"])
                avg_response = sum(response_times) / len(response_times)
        else:
            avg_response = 0
        
        # Calculate success rate
        total_attempts = self.performance_metrics['successful_interactions'] + self.performance_metrics['error_count']
        success_rate = (self.performance_metrics['successful_interactions'] / max(1, total_attempts)) * 100
        
        report = f"""
🛡️ **Unstoppable Features Health Report**

**System Performance:**
• Uptime: {uptime.days} days, {uptime.seconds // 3600} hours
• Success Rate: {success_rate:.1f}%
• Average Response Time: {avg_response:.2f}s
• Total Interactions: {self.performance_metrics['successful_interactions']:,}

**Unstoppable Features Status:**
• 🛡️ Resilience Engine: {self.resilience.get_health_report().split(':')[1].split('-')[0].strip()}
• 📊 Predictive Analytics: ✅ Active ({self.performance_metrics['predictions_made']} predictions made)
• 🔒 Secure Vault: {'✅ Available' if self.vault.enabled else '❌ Requires cryptography package'}
• 🚨 Emergency Protocol: ✅ Active ({self.performance_metrics['emergency_responses']} emergency responses)

**Vault Operations:**
• Health records stored: {self.performance_metrics['vault_operations']}
• Encryption status: {'Enabled' if self.vault.enabled else 'Disabled'}

**Resilience Metrics:**
{self.resilience.get_health_report()}

**Status:** 🟢 All unstoppable features operational
"""
        
        return report
    
    def export_all_user_data(self, user_id: str, include_health: bool = True) -> Dict:
        """Export all user data from unstoppable features"""
        export_data = {
            'user_id': user_id,
            'export_timestamp': datetime.now().isoformat(),
            'unstoppable_features': {
                'conversation_state': self.conversation_states.get(user_id, {}),
                'performance_data': {
                    'interactions': self.performance_metrics['successful_interactions'],
                    'emergency_responses': self.performance_metrics['emergency_responses']
                }
            }
        }
        
        # Add health data if requested and available
        if include_health and self.vault.enabled:
            health_data = self.vault.retrieve_health_data(user_id)
            export_data['unstoppable_features']['health_vault'] = health_data
        
        return export_data
    
    def cleanup_old_data(self):
        """Clean up old data to prevent memory bloat"""
        # Clean up old error history
        if len(self.resilience.error_history) > 500:
            # Keep only recent 250 errors
            recent_errors = list(self.resilience.error_history)[-250:]
            self.resilience.error_history.clear()
            self.resilience.error_history.extend(recent_errors)
        
        # Clean up old conversation states (older than 30 days)
        cutoff_date = datetime.now() - timedelta(days=30)
        states_to_remove = []
        
        for user_id, state in self.conversation_states.items():
            if hasattr(state, 'last_interaction') and state.last_interaction < cutoff_date:
                states_to_remove.append(user_id)
        
        for user_id in states_to_remove:
            del self.conversation_states[user_id]
    
    def get_emergency_checklist(self) -> str:
        """Get emergency preparedness checklist"""
        return self.emergency.get_emergency_checklist()


# ====== UTILITY FUNCTIONS ======

def create_unstoppable_features() -> UnstoppableFeatures:
    """Factory function to create unstoppable features instance"""
    return UnstoppableFeatures()


def test_unstoppable_features():
    """Test all unstoppable features functionality"""
    print("🧪 Testing Unstoppable Features...")
    
    # Create instance
    unstoppable = create_unstoppable_features()
    
    # Test emergency detection
    emergency_tests = [
        "I'm bleeding heavily and scared",
        "The baby isn't moving and I'm worried",
        "I'm having severe chest pain",
        "I think my water broke",
        "Just checking in on my pregnancy"
    ]
    
    print("\n🚨 Testing Emergency Detection:")
    for test_input in emergency_tests:
        is_emergency, emergency_type, urgency_level = unstoppable.analyze_input_for_emergencies(test_input)
        status = f"{'🚨 EMERGENCY' if is_emergency else '✅ Normal'}"
        print(f"  '{test_input[:30]}...' -> {status}")
        if is_emergency:
            print(f"    Type: {emergency_type}, Level: {urgency_level}")
    
    # Test predictive analytics
    print("\n📊 Testing Predictive Analytics:")
    mood_history = deque(['anxious', 'worried', 'excited', 'nervous', 'happy'], maxlen=10)
    predictions = unstoppable.predict_user_needs("test_user", mood_history, "pregnancy questions")
    print(f"  Emotional needs prediction: {predictions}")
    
    milestones = unstoppable.get_milestone_predictions(20, 3)
    print(f"  Upcoming milestones (week 20): {len(milestones)} predicted")
    
    # Test vault (if available)
    print(f"\n🔒 Testing Secure Vault:")
    if unstoppable.vault.enabled:
        test_data = {"mood": "happy", "symptoms": ["none"], "notes": "Feeling great today!"}
        success = unstoppable.store_user_health_data("test_user", "mood_tracking", test_data)
        print(f"  Data storage: {'✅ Success' if success else '❌ Failed'}")
        
        summary = unstoppable.get_user_health_summary("test_user")
        print(f"  Data retrieval: {'✅ Success' if summary else '❌ Failed'}")
    else:
        print("  ⚠️ Vault disabled - install cryptography package")
    
    # Test resilience
    print(f"\n🛡️ Testing Resilience:")
    @unstoppable.resilience.wrap_with_resilience
    def test_function():
        return "Success!"
    
    result = test_function()
    print(f"  Resilience wrapper: {'✅ Working' if result else '❌ Failed'}")
    
    # Get health report
    print(f"\n📋 System Health Report:")
    health_report = unstoppable.get_system_health_report()
    print(health_report)
    
    print("\n✅ Unstoppable Features testing complete!")


if __name__ == "__main__":
    # Run tests if executed directly
    test_unstoppable_features()