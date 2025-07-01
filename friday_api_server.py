# =====================================
# File: friday_api_server.py
# Purpose: Professional FridayAI API Server for Production
# Usage: python friday_api_server.py
# =====================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import json
from typing import Optional, Dict, List

# Import your actual FridayAI
try:
    from fridayai import FridayAI
    from core.MemoryCore import MemoryCore
    from core.EmotionCoreV2 import EmotionCoreV2
    FRIDAY_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ FridayAI modules not found: {e}")
    FRIDAY_AVAILABLE = False

# =====================================
# API Models
# =====================================

class ChatRequest(BaseModel):
    message: str
    user_id: str = "pregnancy_user"
    pregnancy_week: int = 20
    mood: str = "peaceful"
    personality: str = "caring_companion"
    context: Dict = {}

class ChatResponse(BaseModel):
    response: str
    detected_mood: str
    confidence: float
    suggestions: List[str]
    week_info: Dict
    friday_status: str

class SystemStatus(BaseModel):
    status: str
    friday_connected: bool
    features: Dict
    uptime: str
    chat_count: int

# =====================================
# Professional FridayAI API
# =====================================

class FridayAIAPI:
    def __init__(self):
        self.app = FastAPI(
            title="FridayAI Pregnancy Assistant API",
            description="Professional AI pregnancy support powered by FridayAI",
            version="2.0.0"
        )
        
        # Initialize FridayAI
        self.friday_ai = None
        self.chat_count = 0
        
        if FRIDAY_AVAILABLE:
            try:
                print("🧠 Initializing FridayAI brain...")
                memory = MemoryCore(memory_file="friday_memory.enc", key_file="memory.key")
                emotion = EmotionCoreV2()
                self.friday_ai = FridayAI(memory, emotion)
                print("✅ FridayAI brain connected successfully!")
            except Exception as e:
                print(f"❌ Error initializing FridayAI: {e}")
                self.friday_ai = None
        
        self.setup_middleware()
        self.setup_routes()
    
    def setup_middleware(self):
        """Setup CORS and other middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure for your domain in production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.post("/api/v2/chat", response_model=ChatResponse)
        async def process_pregnancy_chat(request: ChatRequest):
            """Main chat endpoint using real FridayAI"""
            return await self.handle_chat(request)
        
        @self.app.get("/api/v2/status", response_model=SystemStatus)
        async def get_system_status():
            """Get comprehensive system status"""
            return self.get_status()
        
        @self.app.get("/api/v2/health")
        async def health_check():
            """Simple health check"""
            return {"status": "healthy", "friday_connected": self.friday_ai is not None}
        
        @self.app.get("/")
        async def root():
            """API information"""
            return {
                "name": "FridayAI Pregnancy Assistant API",
                "version": "2.0.0",
                "status": "running",
                "friday_connected": self.friday_ai is not None,
                "endpoints": {
                    "chat": "POST /api/v2/chat",
                    "status": "GET /api/v2/status",
                    "health": "GET /api/v2/health"
                }
            }
    
    async def handle_chat(self, request: ChatRequest) -> ChatResponse:
        """Handle chat using real FridayAI brain"""
        try:
            self.chat_count += 1
            
            # Pregnancy data
            pregnancy_data = {
                4: {"size": "poppy seed", "emoji": "🌱", "milestone": "Neural tube forming"},
                8: {"size": "raspberry", "emoji": "🫐", "milestone": "Heart beating"},
                12: {"size": "lime", "emoji": "🟢", "milestone": "Major organs formed"},
                16: {"size": "avocado", "emoji": "🥑", "milestone": "First movements"},
                20: {"size": "banana", "emoji": "🍌", "milestone": "Can hear your voice"},
                24: {"size": "corn", "emoji": "🌽", "milestone": "Hearing developing"},
                28: {"size": "eggplant", "emoji": "🍆", "milestone": "Brain developing"},
                32: {"size": "coconut", "emoji": "🥥", "milestone": "Bones hardening"},
                36: {"size": "lettuce", "emoji": "🥬", "milestone": "Almost full-term"},
                40: {"size": "watermelon", "emoji": "🍉", "milestone": "Ready to be born!"}
            }
            
            # Get week info
            closest_week = min(pregnancy_data.keys(), key=lambda x: abs(x - request.pregnancy_week))
            week_info = pregnancy_data[closest_week]
            
            if self.friday_ai:
                # =====================================
                # USE REAL FRIDAYAI BRAIN
                # =====================================
                
                # Set pregnancy context in FridayAI
                self.friday_ai.pregnancy_week = request.pregnancy_week
                self.friday_ai.current_user_id = request.user_id
                self.friday_ai.user_mood = request.mood
                
                # Set personality/tone
                personality_mapping = {
                    'caring_companion': 'supportive',
                    'wise_guide': 'clinical',
                    'cheerful_friend': 'friendly'
                }
                
                if hasattr(self.friday_ai, 'tone_manager'):
                    self.friday_ai.tone_manager.current_tone = personality_mapping.get(
                        request.personality, 'supportive'
                    )
                
                # Create enhanced pregnancy context message
                context_message = f"""
                [PREGNANCY CONTEXT]
                Week: {request.pregnancy_week}
                Baby size: {week_info['size']} {week_info['emoji']}
                Milestone: {week_info['milestone']}
                User mood: {request.mood}
                Personality: {request.personality}
                
                User message: {request.message}
                
                Please respond as a caring, knowledgeable pregnancy AI assistant. Be supportive, 
                medically accurate, and emotionally intelligent. Address their specific week and mood.
                """
                
                # Call your actual FridayAI
                friday_response = self.friday_ai.respond_to(context_message, request.pregnancy_week)
                
                # Process FridayAI response
                if isinstance(friday_response, dict):
                    ai_response = friday_response.get('content', str(friday_response))
                    detected_emotion = friday_response.get('emotional_tone', request.mood)
                    confidence = friday_response.get('confidence', 0.95)
                else:
                    ai_response = str(friday_response)
                    detected_emotion = request.mood
                    confidence = 0.90
                
                friday_status = "real_ai_connected"
                
            else:
                # Fallback response
                ai_response = f"I'm having trouble connecting to my full AI brain right now, but I'm still here for you! At week {request.pregnancy_week}, your baby is the size of a {week_info['size']} {week_info['emoji']} and {week_info['milestone'].lower()}. How can I support you today?"
                detected_emotion = request.mood
                confidence = 0.70
                friday_status = "fallback_mode"
            
            # Generate contextual suggestions
            suggestions = self.generate_suggestions(request.pregnancy_week, detected_emotion)
            
            return ChatResponse(
                response=ai_response,
                detected_mood=detected_emotion,
                confidence=confidence,
                suggestions=suggestions[:3],
                week_info=week_info,
                friday_status=friday_status
            )
            
        except Exception as e:
            print(f"❌ Chat error: {str(e)}")
            
            # Graceful error handling
            return ChatResponse(
                response="I'm experiencing a technical issue, but I'm still here to support you. How are you feeling today?",
                detected_mood="supportive",
                confidence=0.60,
                suggestions=["Tell me about your day", "How is baby doing?", "I need encouragement"],
                week_info=week_info if 'week_info' in locals() else {"size": "growing", "emoji": "💕", "milestone": "developing beautifully"},
                friday_status="error_recovery"
            )
    
    def generate_suggestions(self, week: int, mood: str) -> List[str]:
        """Generate contextual suggestions"""
        suggestions = []
        
        if mood == 'anxious':
            suggestions = [
                "Tell me breathing exercises for pregnancy",
                "What's normal to worry about right now?",
                "Help me feel more confident",
                "Share reassuring pregnancy facts"
            ]
        elif mood == 'excited':
            suggestions = [
                f"What should I expect in week {week + 1}?",
                "Tell me amazing baby development facts",
                "What milestones are coming up?",
                "Share something wonderful about pregnancy"
            ]
        elif mood == 'tired':
            suggestions = [
                "Tips for better pregnancy sleep",
                "How to boost energy naturally",
                "What's normal for fatigue?",
                "Self-care ideas for expectant mothers"
            ]
        else:
            suggestions = [
                f"How is my baby developing at week {week}?",
                "What should I focus on for health?",
                "Tell me something interesting about pregnancy",
                "What symptoms are normal right now?"
            ]
        
        # Add week-specific suggestions
        if week < 13:
            suggestions.append("First trimester guidance")
        elif week < 27:
            suggestions.append("Second trimester what to expect")
        else:
            suggestions.append("Third trimester preparation")
        
        return suggestions
    
    def get_status(self) -> SystemStatus:
        """Get comprehensive system status"""
        features = {
            "real_fridayai": self.friday_ai is not None,
            "pregnancy_knowledge": True,
            "mood_detection": True,
            "contextual_responses": True,
            "week_tracking": True
        }
        
        if self.friday_ai:
            features.update({
                "legendary_memory": hasattr(self.friday_ai, 'legendary_memory'),
                "emotion_core": hasattr(self.friday_ai, 'emotion_core'),
                "goal_coaching": hasattr(self.friday_ai, 'goal_coach'),
                "tone_management": hasattr(self.friday_ai, 'tone_manager')
            })
        
        return SystemStatus(
            status="running",
            friday_connected=self.friday_ai is not None,
            features=features,
            uptime="Active",
            chat_count=self.chat_count
        )

# =====================================
# Server Launch
# =====================================

def main():
    """Launch the professional FridayAI API server"""
    print("=" * 60)
    print("🚀 FRIDAYAI PREGNANCY ASSISTANT API SERVER")
    print("=" * 60)
    
    # Create API instance
    friday_api = FridayAIAPI()
    
    print(f"\n✅ API Server initialized")
    print(f"🧠 FridayAI Status: {'Connected' if friday_api.friday_ai else 'Fallback Mode'}")
    print(f"\n🌐 Starting API server...")
    print(f"📍 API Base URL: http://localhost:8000")
    print(f"📊 API Docs: http://localhost:8000/docs")
    print(f"💬 Chat Endpoint: POST http://localhost:8000/api/v2/chat")
    print(f"\n💡 Ready for CodeSandbox integration!")
    print(f"🔄 Press Ctrl+C to stop\n")
    
    # Launch server
    try:
        uvicorn.run(
            friday_api.app, 
            host="0.0.0.0",  # Allow external connections
            port=8000,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 FridayAI API Server shutting down...")

if __name__ == "__main__":
    main()