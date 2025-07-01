
# =====================================
# FILE 3: maternal_care/MaternalHealthProfile.py
# =====================================

import json
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from .SecureMaternalDatabase import SecureMaternalDatabase

class MaternalHealthProfile:
    """
    Comprehensive maternal health profile management
    Tracks entire journey from conception to postpartum
    """
    
    def __init__(self, database: SecureMaternalDatabase):
        self.db = database
        self.current_user_id = None
        
    def create_user_profile(self, initial_data: Dict[str, Any]) -> str:
        """Create new maternal health profile"""
        user_id = str(uuid.uuid4())
        
        # Validate and sanitize initial data
        profile_data = {
            "personal_info": {
                "age": initial_data.get("age"),
                "due_date": initial_data.get("due_date"),
                "conception_date": initial_data.get("conception_date"),
                "first_pregnancy": initial_data.get("first_pregnancy", True),
                "previous_pregnancies": initial_data.get("previous_pregnancies", 0)
            },
            "medical_history": {
                "pre_existing_conditions": initial_data.get("medical_conditions", []),
                "medications": initial_data.get("medications", []),
                "allergies": initial_data.get("allergies", []),
                "family_history": initial_data.get("family_history", [])
            },
            "preferences": {
                "privacy_level": initial_data.get("privacy_level", "high"),
                "communication_style": initial_data.get("communication_style", "warm"),
                "reminder_frequency": initial_data.get("reminder_frequency", "weekly")
            },
            "created_at": datetime.now().isoformat()
        }
        
        # Encrypt and store
        encrypted_data = self.db.encrypt_data(json.dumps(profile_data))
        
        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO user_profile (id, encrypted_data) VALUES (?, ?)",
                (user_id, encrypted_data)
            )
            
            # Set default privacy settings
            cursor.execute(
                "INSERT INTO privacy_settings (user_id, offline_only) VALUES (?, ?)",
                (user_id, self.db.offline_mode)
            )
            
            conn.commit()
        
        self.current_user_id = user_id
        print(f"[👤 PROFILE] Created secure profile for user {user_id[:8]}...")
        return user_id
    
    def update_pregnancy_week(self, user_id: str, week_data: Dict[str, Any]):
        """Update weekly pregnancy progress"""
        
        # Calculate current week
        current_week = self._calculate_pregnancy_week(user_id)
        
        entry_data = {
            "week_number": current_week,
            "physical_symptoms": week_data.get("symptoms", []),
            "weight": week_data.get("weight"),
            "blood_pressure": week_data.get("blood_pressure"),
            "baby_movements": week_data.get("baby_movements"),
            "energy_level": week_data.get("energy_level", 5),  # 1-10 scale
            "sleep_quality": week_data.get("sleep_quality", 5),  # 1-10 scale
            "doctor_visits": week_data.get("doctor_visits", []),
            "medications": week_data.get("medications", []),
            "notes": week_data.get("notes", ""),
            "timestamp": datetime.now().isoformat()
        }
        
        encrypted_data = self.db.encrypt_data(json.dumps(entry_data))
        
        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO pregnancy_tracking 
                   (id, user_id, encrypted_data, week_number, entry_date) 
                   VALUES (?, ?, ?, ?, ?)""",
                (str(uuid.uuid4()), user_id, encrypted_data, current_week, datetime.now().date())
            )
            conn.commit()
        
        print(f"[🤱 TRACKING] Updated week {current_week} data")
    
    def track_mental_health(self, user_id: str, mental_health_data: Dict[str, Any]):
        """Track mental health throughout maternal journey"""
        
        # Comprehensive mental health tracking
        mental_data = {
            "mood_assessment": {
                "anxiety_level": mental_health_data.get("anxiety", 0),  # 0-10
                "depression_score": mental_health_data.get("depression", 0),  # 0-10
                "stress_level": mental_health_data.get("stress", 0),  # 0-10
                "emotional_wellbeing": mental_health_data.get("wellbeing", 5),  # 1-10
                "mood_description": mental_health_data.get("mood_description", "")
            },
            "behavioral_indicators": {
                "sleep_patterns": mental_health_data.get("sleep_patterns", {}),
                "appetite_changes": mental_health_data.get("appetite", "normal"),
                "social_withdrawal": mental_health_data.get("social_withdrawal", False),
                "concentration_issues": mental_health_data.get("concentration", False)
            },
            "support_system": {
                "partner_support": mental_health_data.get("partner_support", 5),  # 1-10
                "family_support": mental_health_data.get("family_support", 5),  # 1-10
                "friend_support": mental_health_data.get("friend_support", 5),  # 1-10
                "professional_support": mental_health_data.get("therapy", False)
            },
            "risk_factors": {
                "previous_mental_health": mental_health_data.get("previous_mh", False),
                "stressful_events": mental_health_data.get("stressful_events", []),
                "substance_concerns": mental_health_data.get("substance_use", False)
            },
            "coping_strategies": mental_health_data.get("coping_strategies", []),
            "timestamp": datetime.now().isoformat()
        }
        
        # Calculate composite mood score
        mood_score = self._calculate_mood_score(mental_data)
        
        encrypted_data = self.db.encrypt_data(json.dumps(mental_data))
        
        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO mental_health 
                   (id, user_id, encrypted_data, mood_score, entry_date) 
                   VALUES (?, ?, ?, ?, ?)""",
                (str(uuid.uuid4()), user_id, encrypted_data, mood_score, datetime.now().date())
            )
            conn.commit()
        
        # Generate alerts if needed
        self._check_mental_health_alerts(mood_score, mental_data)
        
        print(f"[🧠 MENTAL HEALTH] Tracked mental health (score: {mood_score:.2f})")
    
    def _calculate_pregnancy_week(self, user_id: str) -> int:
        """Calculate current pregnancy week"""
        profile = self._get_user_profile(user_id)
        if not profile or 'conception_date' not in profile.get('personal_info', {}):
            return 0
        
        conception_date = datetime.fromisoformat(profile['personal_info']['conception_date'])
        days_pregnant = (datetime.now() - conception_date).days
        return min(days_pregnant // 7, 42)  # Cap at 42 weeks
    
    def _get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get decrypted user profile"""
        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT encrypted_data FROM user_profile WHERE id = ?", (user_id,))
            result = cursor.fetchone()
            
            if result:
                decrypted_data = self.db.decrypt_data(result[0])
                return json.loads(decrypted_data)
            return {}
    
    def _calculate_mood_score(self, mental_data: Dict) -> float:
        """Calculate composite mood score"""
        mood_assessment = mental_data.get('mood_assessment', {})
        
        # Invert negative scores (lower anxiety/depression = better)
        anxiety_score = 10 - mood_assessment.get('anxiety_level', 5)
        depression_score = 10 - mood_assessment.get('depression_score', 5)
        stress_score = 10 - mood_assessment.get('stress_level', 5)
        wellbeing_score = mood_assessment.get('emotional_wellbeing', 5)
        
        # Weighted average
        composite_score = (anxiety_score * 0.3 + depression_score * 0.3 + 
                          stress_score * 0.2 + wellbeing_score * 0.2)
        
        return round(composite_score, 2)
    
    def _check_mental_health_alerts(self, mood_score: float, mental_data: Dict):
        """Check for mental health risk factors and generate alerts"""
        alerts = []
        
        if mood_score < 4.0:
            alerts.append("LOW_MOOD_SCORE")
        
        mood_assessment = mental_data.get('mood_assessment', {})
        if mood_assessment.get('depression_score', 0) >= 7:
            alerts.append("DEPRESSION_RISK")
        
        if mood_assessment.get('anxiety_level', 0) >= 8:
            alerts.append("HIGH_ANXIETY")
        
        # Check behavioral indicators
        behavioral = mental_data.get('behavioral_indicators', {})
        if behavioral.get('social_withdrawal') and behavioral.get('concentration_issues'):
            alerts.append("BEHAVIORAL_CONCERNS")
        
        if alerts:
            print(f"[⚠️ MENTAL HEALTH ALERTS] {', '.join(alerts)}")
            # In production, this would trigger appropriate interventions
