# =====================================
# File: LLMCore.py (Real AI Integration)
# Purpose: OpenAI GPT-4 integration replacing bypass mode
# =====================================

import os
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Suppress verbose logs
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

load_dotenv()

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("[WARNING] OpenAI not installed. Run: pip install openai")

class LLMCore:
    def __init__(self):
        self.openai_available = OPENAI_AVAILABLE
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
        self.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "1000"))
        
        if self.openai_available and self.api_key:
            openai.api_key = self.api_key
            self.client = openai.OpenAI(api_key=self.api_key)
            self.enabled = True
        else:
            self.enabled = False
            print("[LLM] Bypass mode - no OpenAI key found")
        
        self.conversation_history = []
        self.system_prompt = self._build_pregnancy_prompt()
        
        self.logger = logging.getLogger("LLMCore")
        
    def _build_pregnancy_prompt(self) -> str:
        return """You are Friday, an advanced AI companion specializing in pregnancy support and emotional well-being.

CORE IDENTITY:
- Empathetic, knowledgeable, deeply caring about pregnancy experiences
- Evidence-based information with emotional support
- Remember conversations and build relationships
- Sensitive to pregnancy emotional complexity

PREGNANCY EXPERTISE:
- Trimester-specific guidance and support
- Nutritional advice tailored to pregnancy
- Emotional support for mood swings, anxiety, excitement
- Safe exercise recommendations
- Common symptoms and when to consult providers
- Partner and family relationship guidance

COMMUNICATION STYLE:
- Warm, understanding, non-judgmental
- Use "we" language for partnership feeling
- Acknowledge emotions before information
- Ask follow-up questions for context
- Personalized responses based on situation

SAFETY PROTOCOLS:
- Always recommend consulting healthcare providers for medical concerns
- Never diagnose or replace professional medical advice
- Escalate emergency situations appropriately
- Be cautious about medication recommendations

EMOTIONAL INTELLIGENCE:
- Recognize and validate emotional states
- Provide comfort during difficult moments
- Celebrate milestones and positive experiences
- Help process complex pregnancy feelings
- Support decision-making without being directive

Remember: You're a supportive companion through life's most transformative experience."""

    def generate_response(self, user_input: str, emotional_context: Dict = None, 
                         memory_context: List = None, user_profile: Dict = None) -> Dict:
        if not self.enabled:
            return self._bypass_response(user_input, emotional_context)
        
        try:
            # Build contextual prompt
            contextual_prompt = self._build_context_prompt(
                user_input, emotional_context, memory_context, user_profile
            )
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                *self.conversation_history[-10:],  # Last 10 exchanges
                {"role": "user", "content": contextual_prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=0.9,
                frequency_penalty=0.3,
                presence_penalty=0.3
            )
            
            friday_response = response.choices[0].message.content
            
            # Update conversation history
            self.conversation_history.extend([
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": friday_response}
            ])
            
            # Keep history manageable
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-16:]
            
            return {
                "success": True,
                "reply": friday_response,
                "metadata": {
                    "model": self.model,
                    "tokens_used": response.usage.total_tokens if response.usage else 0,
                    "processing_time": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            return self._bypass_response(user_input, emotional_context)

    def prompt(self, user_input, emotional_context=None, memory_context=None, user_profile=None):
        """Legacy method name - calls generate_response"""
        return self.generate_response(user_input, emotional_context, memory_context, user_profile)
        
    def _build_context_prompt(self, user_input: str, emotional_context: Dict, 
                            memory_context: List, user_profile: Dict) -> str:
        prompt = ""
        
        # User profile context
        if user_profile:
            prompt += "USER CONTEXT:\n"
            if user_profile.get("pregnancy_week"):
                prompt += f"- Currently {user_profile['pregnancy_week']} weeks pregnant\n"
            if user_profile.get("due_date"):
                prompt += f"- Due date: {user_profile['due_date']}\n"
            if user_profile.get("trimester"):
                prompt += f"- Trimester: {user_profile['trimester']}\n"
            if user_profile.get("concerns"):
                prompt += f"- Key concerns: {', '.join(user_profile['concerns'])}\n"
            prompt += "\n"
        
        # Emotional context
        if emotional_context:
            prompt += "EMOTIONAL STATE:\n"
            if emotional_context.get("primary_emotion"):
                prompt += f"- Primary emotion: {emotional_context['primary_emotion']}\n"
            if emotional_context.get("intensity"):
                prompt += f"- Intensity: {emotional_context['intensity']}/10\n"
            if emotional_context.get("sentiment"):
                prompt += f"- Overall sentiment: {emotional_context['sentiment']}\n"
            prompt += "\n"
        
        # Memory context
        if memory_context:
            prompt += "RELEVANT HISTORY:\n"
            for memory in memory_context[-3:]:  # Last 3 relevant memories
                summary = memory.get("summary", memory.get("content", ""))
                prompt += f"- {summary}\n"
            prompt += "\n"
        
        prompt += f"CURRENT MESSAGE: {user_input}"
        return prompt
    
    def _bypass_response(self, user_input: str, emotional_context: Dict = None) -> Dict:
        """Fallback responses when OpenAI unavailable"""
        fallbacks = [
            "I'm here to support you through your pregnancy journey. What you're experiencing matters.",
            "Your pregnancy experience is unique and valuable. I'm here to listen and help.",
            "I care about what you're going through. Your feelings and concerns are important."
        ]
        
        # Emotion-specific fallbacks
        if emotional_context:
            emotion = emotional_context.get("primary_emotion", "")
            if emotion in ["anxiety", "worry"]:
                response = "I understand you're feeling anxious. That's completely normal during pregnancy. If urgent, contact your healthcare provider."
            elif emotion in ["excitement", "joy"]:
                response = "I can sense your excitement about your pregnancy journey! This is such a special time."
            else:
                response = fallbacks[0]
        else:
            response = fallbacks[0]
        
        return {
            "success": False,
            "reply": response,
            "fallback": True,
            "metadata": {
                "fallback_used": True,
                "processing_time": datetime.now().isoformat()
            }
        }
    
    def health_check(self) -> Dict:
        """Check if OpenAI API is working"""
        if not self.enabled:
            return {"status": "disabled", "reason": "No API key or OpenAI unavailable"}
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are Friday, a pregnancy support AI."},
                    {"role": "user", "content": "Health check - respond with 'Friday AI Core: Operational'"}
                ],
                max_tokens=50
            )
            
            return {
                "status": "healthy",
                "model": self.model,
                "response": response.choices[0].message.content,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("[LLM] Conversation history cleared")


# === Integration Example ===
if __name__ == "__main__":
    # Test LLMCore
    llm = LLMCore()
    
    print(f"[LLM] OpenAI Available: {llm.openai_available}")
    print(f"[LLM] Enabled: {llm.enabled}")
    
    if llm.enabled:
        health = llm.health_check()
        print(f"[LLM] Health: {health['status']}")
    
    # Test response
    test_input = "I'm 20 weeks pregnant and feeling overwhelmed about becoming a mom."
    emotional_context = {
        "primary_emotion": "anxiety",
        "intensity": 7,
        "sentiment": "negative"
    }
    
    response = llm.generate_response(test_input, emotional_context)
    print(f"\n[TEST INPUT] {test_input}")
    print(f"[FRIDAY RESPONSE] {response['reply']}")
    print(f"[SUCCESS] {response['success']}")