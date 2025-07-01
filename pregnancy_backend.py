# =====================================
# File: pregnancy_backend.py
# Purpose: Standalone pregnancy assistant backend that can import FridayAI
# Usage: python pregnancy_backend.py (runs independently)
#        OR: from pregnancy_backend import PregnancyBackend (import in main)
# =====================================

import logging
import os
import sys
import json
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Suppress verbose logs for clean output
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# =============================================================
# 🌐 API MODELS
# =============================================================

class ChatRequest(BaseModel):
    message: str
    context: dict

class ChatResponse(BaseModel):
    response: str
    detectedMood: str
    confidence: float
    suggestions: List[str]

# =============================================================
# 🤱 PREGNANCY KNOWLEDGE BASE
# =============================================================

PREGNANCY_WEEKS_DATA = {
    4: {"size": "poppy seed", "emoji": "🌱", "milestone": "Your baby's neural tube is forming"},
    8: {"size": "raspberry", "emoji": "🫐", "milestone": "Baby's heart is beating and limbs are developing"},
    12: {"size": "lime", "emoji": "🟢", "milestone": "All major organs have formed"},
    16: {"size": "avocado", "emoji": "🥑", "milestone": "You might feel first movements soon"},
    20: {"size": "banana", "emoji": "🍌", "milestone": "Halfway point! Baby can hear your voice"},
    24: {"size": "ear of corn", "emoji": "🌽", "milestone": "Baby's hearing is developing rapidly"},
    28: {"size": "eggplant", "emoji": "🍆", "milestone": "Baby's brain is developing quickly"},
    32: {"size": "jicama", "emoji": "🥥", "milestone": "Baby's bones are hardening"},
    36: {"size": "romaine lettuce", "emoji": "🥬", "milestone": "Baby is considered full-term soon"},
    40: {"size": "watermelon", "emoji": "🍉", "milestone": "Baby is ready to meet you!"}
}

def get_week_info(week: int) -> Dict:
    """Get pregnancy information for specific week"""
    # Find closest week data
    closest_week = min(PREGNANCY_WEEKS_DATA.keys(), key=lambda x: abs(x - week))
    base_info = PREGNANCY_WEEKS_DATA[closest_week].copy()
    
    # Adjust for exact week if needed
    if week != closest_week:
        if week < 12:
            base_info["milestone"] = f"Early development continues (week {week})"
        elif week < 28:
            base_info["milestone"] = f"Second trimester growth (week {week})"
        else:
            base_info["milestone"] = f"Final preparations for birth (week {week})"
    
    return base_info

# =============================================================
# 🧠 PREGNANCY BACKEND CLASS
# =============================================================

class PregnancyBackend:
    def __init__(self, friday_ai_instance=None):
        """
        Initialize pregnancy backend
        
        Args:
            friday_ai_instance: Optional FridayAI instance to use real brain
                               If None, uses enhanced placeholder responses
        """
        self.friday_ai = friday_ai_instance
        self.app = FastAPI(
            title="FridayAI Pregnancy Assistant", 
            version="1.0.0",
            description="Intelligent pregnancy support powered by FridayAI"
        )
        
        self.setup_cors()
        self.setup_routes()
        self.server_thread = None
        self.server_running = False
        
        # Performance tracking
        self.chat_count = 0
        self.start_time = datetime.now()
        
        print("🤱 Pregnancy Backend initialized")
        if self.friday_ai:
            print("🧠 Connected to FridayAI brain")
        else:
            print("💡 Using enhanced standalone mode (no FridayAI connection)")
    
    def setup_cors(self):
        """Setup CORS for React frontend"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.post("/api/chat", response_model=ChatResponse)
        async def pregnancy_chat(request: ChatRequest):
            return await self.handle_pregnancy_chat(request)
        
        @self.app.get("/api/friday_status")
        async def get_friday_status():
            return self.get_system_status()
        
        @self.app.get("/api/health")
        async def health_check():
            return {
                "status": "healthy",
                "uptime": str(datetime.now() - self.start_time),
                "chat_count": self.chat_count
            }
        
        @self.app.get("/")
        async def root():
            return {
                "message": "🤱 FridayAI Pregnancy Assistant Backend",
                "status": "running",
                "friday_connected": self.friday_ai is not None,
                "endpoints": [
                    "POST /api/chat",
                    "GET /api/friday_status", 
                    "GET /api/health"
                ]
            }
    
    async def handle_pregnancy_chat(self, request: ChatRequest) -> ChatResponse:
        """Handle pregnancy chat requests"""
        try:
            self.chat_count += 1
            message = request.message
            context = request.context
            
            # Extract context information
            pregnancy_week = context.get('pregnancyWeek', 20)
            current_mood = context.get('currentMood', 'peaceful')
            personality = context.get('personality', 'caring_companion')
            user_id = context.get('userId', 'pregnancy_user')
            
            print(f"💬 Chat #{self.chat_count}: Week {pregnancy_week}, Mood: {current_mood}")
            
            # =============================================================
            # 🧠 USE FRIDAYAI BRAIN IF AVAILABLE
            # =============================================================
            
            if self.friday_ai:
                # Set FridayAI context
                self.friday_ai.pregnancy_week = pregnancy_week
                self.friday_ai.current_user_id = user_id
                
                # Set tone based on personality
                personality_to_tone = {
                    'caring_companion': 'supportive',
                    'wise_guide': 'clinical', 
                    'cheerful_friend': 'friendly'
                }
                
                if hasattr(self.friday_ai, 'tone_manager'):
                    self.friday_ai.tone_manager.current_tone = personality_to_tone.get(personality, 'supportive')
                
                # Create enhanced message with pregnancy context
                enhanced_message = f"[Pregnancy Week {pregnancy_week}, Mood: {current_mood}] {message}"
                
                # Call FridayAI respond_to method
                friday_response = self.friday_ai.respond_to(enhanced_message, pregnancy_week)
                
                # Extract response content
                if isinstance(friday_response, dict):
                    ai_response = friday_response.get('content', str(friday_response))
                    detected_emotion = friday_response.get('emotional_tone', current_mood)
                    confidence = friday_response.get('confidence', 0.95)
                else:
                    ai_response = str(friday_response)
                    detected_emotion = current_mood
                    confidence = 0.95
                
                print("✅ FridayAI brain processed request")
            
            else:
                # =============================================================
                # 💡 ENHANCED STANDALONE MODE
                # =============================================================
                
                ai_response = await self.generate_smart_pregnancy_response(
                    message, pregnancy_week, current_mood, personality
                )
                detected_emotion = self.detect_mood_from_message(message, current_mood)
                confidence = 0.85
                
                print("✅ Standalone mode processed request")
            
            # Generate contextual suggestions
            suggestions = self.generate_pregnancy_suggestions(pregnancy_week, detected_emotion, ai_response)
            
            return ChatResponse(
                response=ai_response,
                detectedMood=detected_emotion,
                confidence=confidence,
                suggestions=suggestions[:3]
            )
            
        except Exception as e:
            print(f"❌ Error in pregnancy chat: {str(e)}")
            
            # Graceful error response
            error_response = self.get_graceful_error_response(message if 'message' in locals() else "")
            
            return ChatResponse(
                response=error_response,
                detectedMood="supportive",
                confidence=0.7,
                suggestions=[
                    "Tell me about breathing exercises",
                    "What's normal during pregnancy?",
                    "I need some encouragement"
                ]
            )
    
    async def generate_smart_pregnancy_response(self, message: str, week: int, mood: str, personality: str) -> str:
        """Generate intelligent pregnancy responses when FridayAI not available"""
        
        message_lower = message.lower()
        week_info = get_week_info(week)
        
        # Personality-based response styles
        if personality == 'caring_companion':
            empathy_prefix = "💙 "
            tone = "warm and nurturing"
        elif personality == 'wise_guide':
            empathy_prefix = "🌟 "
            tone = "knowledgeable and gentle"
        else:  # cheerful_friend
            empathy_prefix = "😊 "
            tone = "friendly and encouraging"
        
        # Emotional pregnancy responses
        if any(word in message_lower for word in ["nervous", "worried", "scared", "anxious"]):
            return f"""{empathy_prefix}I completely understand those feelings - they're so incredibly normal at this stage of your journey. Your little one is about the size of a {week_info['size']} {week_info['emoji']} and {week_info['milestone']}. 

These worries actually show how much you already love your baby. Many expectant mothers experience these same feelings around week {week}. What specific concerns can I help you work through? I'm here to support you every step of the way. ✨

Remember: You're doing an amazing job, and your body knows exactly what it's doing. Trust in this incredible process."""
        
        elif any(word in message_lower for word in ["excited", "happy", "amazing", "wonderful"]):
            return f"""{empathy_prefix}Your excitement just lights up everything! I can feel your joy and it's absolutely beautiful! At week {week}, your baby is the size of a {week_info['size']} {week_info['emoji']} and {week_info['milestone']}.

This excitement you're feeling? Your little one can sense that positive energy too! Your joy is actually beneficial for your baby's development. What's making you happiest about your pregnancy journey right now? I love celebrating these moments with you! 🎉"""
        
        elif any(word in message_lower for word in ["tired", "exhausted", "sleepy"]):
            return f"""{empathy_prefix}Oh sweetheart, growing a human is literally the most incredible and exhausting work in the universe! Your body is working 24/7 to nurture your precious {week_info['size']}-sized baby {week_info['emoji']}.

Right now, {week_info['milestone'].lower()}, which requires enormous energy. That tiredness? It's your body's way of telling you that you're doing something absolutely extraordinary. Rest isn't lazy - it's essential medicine for both you and your baby.

💤 Tips for better rest: Try a pregnancy pillow, take short naps when possible, and remember that every moment of rest is helping your baby grow. How can I help you feel more comfortable? 💕"""
        
        elif any(word in message_lower for word in ["development", "growing", "baby"]):
            return f"""{empathy_prefix}Your baby's development right now is absolutely mind-blowing! At week {week}, they're about the size of a {week_info['size']} {week_info['emoji']} and {week_info['milestone']}.

🌱 Amazing facts for week {week}:
• Your baby's brain is forming 250,000 neurons per minute
• They can already taste what you eat through the amniotic fluid
• Their unique fingerprints are forming
• They're practicing breathing movements

What aspect of your baby's growth are you most curious about? The science of pregnancy never stops amazing me! 💫"""
        
        else:
            return f"""{empathy_prefix}Thank you for sharing that with me - I'm so honored to be part of your pregnancy journey! At week {week}, your experience is truly special. Your little {week_info['size']} {week_info['emoji']} is {week_info['milestone'].lower()}.

You're doing such an amazing job nurturing them. Every day brings new growth and development. I'm here to support you through every moment, every question, every feeling. What's on your heart today? ✨

Remember: You're stronger than you know, and this journey is uniquely yours. Trust in yourself and this incredible process."""
    
    def detect_mood_from_message(self, message: str, current_mood: str) -> str:
        """Enhanced mood detection from message content"""
        message_lower = message.lower()
        
        # Strong emotional indicators
        if any(word in message_lower for word in ["terrified", "panicking", "breakdown"]):
            return "very_anxious"
        elif any(word in message_lower for word in ["scared", "afraid", "nervous", "worried", "anxious"]):
            return "anxious"
        elif any(word in message_lower for word in ["overwhelmed", "stressed", "can't cope"]):
            return "overwhelmed"
        elif any(word in message_lower for word in ["sad", "crying", "depressed", "down"]):
            return "sad"
        elif any(word in message_lower for word in ["thrilled", "ecstatic", "overjoyed"]):
            return "very_excited"
        elif any(word in message_lower for word in ["excited", "happy", "wonderful", "amazing"]):
            return "excited"
        elif any(word in message_lower for word in ["calm", "peaceful", "relaxed"]):
            return "peaceful"
        elif any(word in message_lower for word in ["tired", "exhausted", "drained"]):
            return "tired"
        else:
            return current_mood  # Default to current mood
    
    def generate_pregnancy_suggestions(self, week: int, mood: str, response: str) -> List[str]:
        """Generate contextual pregnancy suggestions"""
        suggestions = []
        
        # Mood-based suggestions
        if mood in ['anxious', 'worried', 'scared', 'very_anxious']:
            suggestions.extend([
                "Tell me about breathing exercises for pregnancy",
                "What's normal to worry about right now?",
                "Help me feel more confident about my pregnancy",
                "Share some reassuring pregnancy facts"
            ])
        elif mood in ['excited', 'happy', 'very_excited']:
            suggestions.extend([
                f"What should I expect in week {week + 1}?",
                "Tell me about my baby's development",
                "What milestones are coming up?",
                "Share something amazing about pregnancy"
            ])
        elif mood in ['tired', 'exhausted']:
            suggestions.extend([
                "Tips for better sleep during pregnancy",
                "How to boost energy naturally",
                "What's normal for fatigue right now?",
                "Self-care ideas for pregnancy"
            ])
        else:
            suggestions.extend([
                f"How is my baby developing at week {week}?",
                "What should I be doing for my health?",
                "Tell me something interesting about pregnancy",
                "What symptoms are normal right now?"
            ])
        
        # Week-specific suggestions
        if week < 13:
            suggestions.append("First trimester tips and advice")
        elif week < 27:
            suggestions.append("Second trimester what to expect")
        else:
            suggestions.append("Third trimester preparation tips")
        
        # Add response-based suggestions
        if "exercise" in response.lower():
            suggestions.append("Safe exercises for my pregnancy stage")
        if "nutrition" in response.lower() or "eating" in response.lower():
            suggestions.append("Healthy pregnancy nutrition tips")
        
        return list(set(suggestions))  # Remove duplicates
    
    def get_graceful_error_response(self, message: str) -> str:
        """Generate graceful error responses"""
        error_responses = [
            "💙 I'm having a little trouble processing that right now, but I'm still here for you. Could you tell me how you're feeling today?",
            "✨ Something's not quite right on my end, but your wellbeing is my priority. What's the most important thing you need support with right now?",
            "🤗 I hit a small snag, but I'm still here to listen and support you. How are you and your baby doing today?",
            "💕 I'm experiencing a minor issue, but you're what matters most. What would be most helpful for you right now?"
        ]
        
        import random
        return random.choice(error_responses)
    
    def get_system_status(self):
        """Get comprehensive system status"""
        try:
            base_status = {
                "status": "running",
                "mode": "friday_ai_connected" if self.friday_ai else "standalone",
                "uptime": str(datetime.now() - self.start_time),
                "chat_count": self.chat_count,
                "features": {
                    "pregnancy_knowledge": True,
                    "mood_detection": True,
                    "contextual_suggestions": True,
                    "graceful_error_handling": True
                }
            }
            
            if self.friday_ai:
                # Add FridayAI specific status
                friday_status = {
                    "friday_ai_features": {
                        "legendary_memory": hasattr(self.friday_ai, 'legendary_memory'),
                        "goal_coaching": hasattr(self.friday_ai, 'goal_coach'),
                        "tone_management": hasattr(self.friday_ai, 'tone_manager'),
                        "unstoppable_features": hasattr(self.friday_ai, 'unstoppable')
                    }
                }
                
                if hasattr(self.friday_ai, 'legendary_memory'):
                    friday_status["conversation_count"] = len(self.friday_ai.legendary_memory.conversations)
                
                if hasattr(self.friday_ai, 'tone_manager'):
                    friday_status["current_tone"] = self.friday_ai.tone_manager.current_tone
                
                base_status.update(friday_status)
            
            return base_status
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "mode": "emergency_fallback"
            }
    
    def start_server(self, host="127.0.0.1", port=8000):
        """Start the pregnancy backend server"""
        try:
            print(f"🚀 Starting FridayAI Pregnancy Backend at http://{host}:{port}")
            print("🌐 Pregnancy web interface: http://127.0.0.1:8000")
            uvicorn.run(self.app, host=host, port=port, log_level="warning")
            return f"🚀 Pregnancy backend started at http://{host}:{port}"
        except Exception as e:
            print(f"❌ Server error: {e}")
            return f"❌ Failed to start server: {e}"
    
    def stop_server(self):
        """Stop the pregnancy backend server"""
        if self.server_running:
            self.server_running = False
            return "🛑 Pregnancy backend stopped"
        return "❌ Pregnancy backend not running"

# =============================================================
# 🚀 STANDALONE EXECUTION
# =============================================================

def main():
    """Run pregnancy backend as standalone application"""
    print("=" * 60)
    print("🤱 FRIDAYAI PREGNANCY ASSISTANT BACKEND")
    print("=" * 60)
    
    # Try to import FridayAI
    friday_ai = None
    try:
        # Attempt to import FridayAI from current directory
        from fridayai import FridayAI
        from core.MemoryCore import MemoryCore
        from core.EmotionCoreV2 import EmotionCoreV2
        
        print("🧠 Initializing FridayAI brain...")
        memory = MemoryCore(memory_file="friday_memory.enc", key_file="memory.key")
        emotion = EmotionCoreV2()
        friday_ai = FridayAI(memory, emotion)
        print("✅ FridayAI brain connected successfully!")
        
    except ImportError as e:
        print(f"⚠️  FridayAI not found: {e}")
        print("💡 Running in enhanced standalone mode")
        friday_ai = None
    except Exception as e:
        print(f"❌ Error initializing FridayAI: {e}")
        print("💡 Running in enhanced standalone mode")
        friday_ai = None
    
    # Create and start pregnancy backend
    backend = PregnancyBackend(friday_ai)
    
    try:
        print("\n🌐 Starting web server...")
        print("🔗 Available endpoints:")
        print("   • POST /api/chat - Pregnancy chat")
        print("   • GET /api/friday_status - System status")
        print("   • GET /api/health - Health check")
        print("   • GET / - API info")
        print("\n💡 Press Ctrl+C to stop the server")
        
        # Start server (blocking)
        uvicorn.run(backend.app, host="127.0.0.1", port=8000, log_level="warning")
        
    except KeyboardInterrupt:
        print("\n\n👋 Pregnancy backend shutting down gracefully...")
        print("✅ Server stopped")

# =============================================================
# 🎯 IMPORT INTERFACE FOR MAIN FRIDAYAI
# =============================================================

def create_pregnancy_backend(friday_ai_instance):
    """
    Create pregnancy backend instance for import in main FridayAI
    
    Usage in your main fridayai.py:
        from pregnancy_backend import create_pregnancy_backend
        pregnancy_backend = create_pregnancy_backend(ai)
        pregnancy_backend.start_server()
    """
    return PregnancyBackend(friday_ai_instance)

if __name__ == "__main__":
    main()