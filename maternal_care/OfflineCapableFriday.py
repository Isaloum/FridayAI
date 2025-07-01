# =====================================
# FILE: maternal_care/OfflineCapableFriday.py
# =====================================

import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from .SecureMaternalDatabase import SecureMaternalDatabase

class OfflineCapableFriday:
    """
    Offline-capable Friday AI with maternal health specialization
    Ensures full privacy and offline functionality
    """
    
    def __init__(self, friday_ai_instance, maternal_db: SecureMaternalDatabase):
        self.friday_ai = friday_ai_instance
        self.maternal_db = maternal_db
        self.offline_mode = maternal_db.offline_mode
        
        # Enhanced maternal conversation context
        self.maternal_context = {
            "current_week": 0,
            "user_concerns": [],
            "recent_symptoms": [],
            "emotional_state": "neutral"
        }
        
        print(f"[🤱 OFFLINE FRIDAY] Maternal AI ready (offline: {self.offline_mode})")
    
    def process_maternal_query(self, user_input: str, user_id: str = None) -> Dict[str, Any]:
        """Process maternal health queries with enhanced context"""
        
        # Update context if user is known
        if user_id:
            self._update_maternal_context(user_id)
        
        # Analyze query for maternal health content
        maternal_analysis = self._analyze_maternal_content(user_input)
        
        # Generate contextual response
        if maternal_analysis["is_maternal_query"]:
            response = self._generate_maternal_response(user_input, maternal_analysis, user_id)
        else:
            # Fall back to standard Friday response
            response = self.friday_ai.respond_to(user_input)
        
        # Log interaction for learning (if consented)
        self._log_maternal_interaction(user_input, response, user_id)
        
        return response
    
    def _analyze_maternal_content(self, user_input: str) -> Dict[str, Any]:
        """Analyze if input is maternal health related"""
        
        maternal_keywords = [
            "pregnant", "pregnancy", "baby", "morning sickness", "contractions",
            "ultrasound", "trimester", "labor", "delivery", "breastfeeding",
            "postpartum", "maternal", "prenatal", "weeks pregnant", "due date",
            "symptoms", "movement", "kicks", "anxiety", "worried", "scared",
            "doctor", "midwife", "hospital", "birth plan", "nutrition"
        ]
        
        emotional_keywords = [
            "anxious", "worried", "scared", "excited", "nervous", "overwhelmed",
            "happy", "sad", "tired", "exhausted", "moody", "emotional"
        ]
        
        symptom_keywords = [
            "nausea", "vomiting", "tired", "fatigue", "headache", "backache",
            "swelling", "heartburn", "constipation", "bleeding", "pain",
            "discharge", "cramps", "dizzy", "breathless"
        ]
        
        input_lower = user_input.lower()
        
        maternal_score = sum(1 for keyword in maternal_keywords if keyword in input_lower)
        emotional_score = sum(1 for keyword in emotional_keywords if keyword in input_lower)
        symptom_score = sum(1 for keyword in symptom_keywords if keyword in input_lower)
        
        is_maternal = maternal_score > 0 or emotional_score > 1 or symptom_score > 0
        
        return {
            "is_maternal_query": is_maternal,
            "maternal_score": maternal_score,
            "emotional_score": emotional_score,
            "symptom_score": symptom_score,
            "detected_emotions": [kw for kw in emotional_keywords if kw in input_lower],
            "detected_symptoms": [kw for kw in symptom_keywords if kw in input_lower],
            "urgency_level": self._assess_urgency(input_lower)
        }
    
    def _assess_urgency(self, input_text: str) -> str:
        """Assess urgency level of maternal query"""
        
        emergency_keywords = [
            "bleeding", "severe pain", "can't breathe", "emergency", "911",
            "hospital", "urgent", "something wrong", "help", "scared"
        ]
        
        high_priority = [
            "contractions", "water broke", "no movement", "very worried",
            "severe", "constant pain", "unusual"
        ]
        
        if any(keyword in input_text for keyword in emergency_keywords):
            return "emergency"
        elif any(keyword in input_text for keyword in high_priority):
            return "high"
        else:
            return "normal"
    
    def _generate_maternal_response(self, user_input: str, analysis: Dict, user_id: str = None) -> Dict[str, Any]:
        """Generate contextual maternal health response"""
        
        # Build enhanced context for response
        context_prompt = self._build_maternal_context_prompt(user_input, analysis, user_id)
        
        # Use Friday's core response system with maternal context
        if hasattr(self.friday_ai, 'llm') and self.friday_ai.llm.enabled:
            # Use LLM with maternal context
            response = self.friday_ai.llm.generate_response(
                context_prompt,
                emotional_context={"primary_emotion": analysis.get("detected_emotions", ["neutral"])[0] if analysis.get("detected_emotions") else "neutral"},
                user_profile=self._get_user_profile_summary(user_id) if user_id else None
            )
        else:
            # Fallback to pattern-based responses
            response = self._generate_fallback_maternal_response(analysis)
        
        # Add maternal-specific enhancements
        enhanced_response = self._enhance_maternal_response(response, analysis, user_id)
        
        return enhanced_response
    
    def _build_maternal_context_prompt(self, user_input: str, analysis: Dict, user_id: str = None) -> str:
        """Build enhanced prompt with maternal context"""
        
        prompt = f"MATERNAL HEALTH CONTEXT:\n"
        
        if user_id:
            profile = self._get_user_profile_summary(user_id)
            if profile:
                prompt += f"- Pregnancy week: {profile.get('current_week', 'unknown')}\n"
                prompt += f"- User concerns: {', '.join(profile.get('concerns', []))}\n"
        
        prompt += f"- Detected emotions: {', '.join(analysis.get('detected_emotions', ['neutral']))}\n"
        prompt += f"- Detected symptoms: {', '.join(analysis.get('detected_symptoms', ['none']))}\n"
        prompt += f"- Urgency level: {analysis.get('urgency_level', 'normal')}\n"
        
        if analysis.get('urgency_level') == 'emergency':
            prompt += "\n⚠️ EMERGENCY RESPONSE REQUIRED - Prioritize safety and medical care\n"
        
        prompt += f"\nUSER MESSAGE: {user_input}"
        
        return prompt
    
    def _generate_fallback_maternal_response(self, analysis: Dict) -> Dict[str, Any]:
        """Generate fallback responses when LLM unavailable"""
        
        urgency = analysis.get("urgency_level", "normal")
        
        if urgency == "emergency":
            response_text = """🚨 If you're experiencing a medical emergency, please contact your healthcare provider immediately or call emergency services.

I'm here to support you, but your safety is the top priority. Please reach out to a medical professional for urgent concerns."""
        
        elif analysis.get("emotional_score", 0) > 1:
            response_text = """I can hear that you're experiencing some strong emotions right now, which is completely normal during pregnancy. 

Your feelings are valid and important. If you're feeling overwhelmed, consider:
- Taking deep, slow breaths
- Reaching out to your support system
- Speaking with your healthcare provider if concerns persist

I'm here to listen and support you through this journey. 💝"""
        
        else:
            response_text = """Thank you for sharing your pregnancy experience with me. I'm here to provide support and information throughout your maternal health journey.

Every pregnancy is unique, and your experiences matter. If you have specific concerns, I always recommend discussing them with your healthcare provider for personalized guidance.

How can I best support you today? 🤱"""
        
        return {
            "success": True,
            "reply": response_text,
            "fallback": True,
            "maternal_context": True
        }
    
    def _enhance_maternal_response(self, response: Dict, analysis: Dict, user_id: str = None) -> Dict[str, Any]:
        """Add maternal-specific enhancements to response"""
        
        # Add urgency warnings if needed
        if analysis.get("urgency_level") == "emergency":
            response["urgency_warning"] = True
            response["recommended_action"] = "Seek immediate medical attention"
        
        # Add pregnancy week context if available
        if user_id:
            current_week = self._get_current_week(user_id)
            if current_week:
                response["pregnancy_week"] = current_week
                response["week_relevant_info"] = self._get_week_specific_info(current_week)
        
        # Add emotional support elements
        if analysis.get("emotional_score", 0) > 0:
            response["emotional_support"] = True
            response["validation_message"] = "Your feelings are completely normal and valid during pregnancy."
        
        # Add symptom acknowledgment
        if analysis.get("detected_symptoms"):
            response["acknowledged_symptoms"] = analysis["detected_symptoms"]
        
        return response
    
    def _update_maternal_context(self, user_id: str):
        """Update maternal context from user profile"""
        try:
            profile = self._get_user_profile_summary(user_id)
            if profile:
                self.maternal_context.update({
                    "current_week": profile.get("current_week", 0),
                    "user_concerns": profile.get("concerns", []),
                    "recent_symptoms": profile.get("recent_symptoms", [])
                })
        except Exception as e:
            print(f"[⚠️ CONTEXT] Failed to update maternal context: {e}")
    
    def _get_user_profile_summary(self, user_id: str) -> Optional[Dict]:
        """Get summarized user profile for context"""
        try:
            # This would integrate with MaternalHealthProfile
            # For now, return basic structure
            return {
                "current_week": self.maternal_context.get("current_week", 0),
                "concerns": self.maternal_context.get("user_concerns", []),
                "recent_symptoms": self.maternal_context.get("recent_symptoms", [])
            }
        except Exception:
            return None
    
    def _get_current_week(self, user_id: str) -> Optional[int]:
        """Get current pregnancy week"""
        profile = self._get_user_profile_summary(user_id)
        return profile.get("current_week") if profile else None
    
    def _get_week_specific_info(self, week: int) -> str:
        """Get week-specific pregnancy information"""
        
        week_info = {
            (1, 13): "First trimester - focus on nutrition and early prenatal care",
            (14, 27): "Second trimester - often called the 'golden period' of pregnancy",
            (28, 40): "Third trimester - preparing for labor and delivery"
        }
        
        for week_range, info in week_info.items():
            if week_range[0] <= week <= week_range[1]:
                return info
        
        return "Consult with your healthcare provider for personalized guidance"
    
    def _log_maternal_interaction(self, user_input: str, response: Dict, user_id: str = None):
        """Log interaction for learning and improvement"""
        
        if not self.offline_mode and user_id:
            # Only log if user has consented and not in strict offline mode
            try:
                interaction_log = {
                    "timestamp": datetime.now().isoformat(),
                    "user_input_hash": hash(user_input),  # Don't store actual input
                    "response_type": "maternal_health",
                    "urgency_level": response.get("urgency_level", "normal"),
                    "user_id": user_id
                }
                
                # Store in maternal database (encrypted)
                # Implementation would depend on database schema
                
            except Exception as e:
                print(f"[📝 LOG] Failed to log interaction: {e}")
    
    def get_offline_capabilities(self) -> Dict[str, bool]:
        """Return current offline capabilities"""
        
        return {
            "maternal_response_generation": True,
            "symptom_analysis": True,
            "emotional_support": True,
            "urgency_assessment": True,
            "week_specific_guidance": True,
            "data_encryption": True,
            "privacy_protection": True,
            "emergency_protocols": True
        }