# FridayAI.py
# True Friday Main Brain with OpenAI Superbrain Bridge

import os
from dotenv import load_dotenv
from MemoryCore import MemoryCore
from EmotionCore import EmotionCore
from KnowledgeCore import KnowledgeCore
import pyttsx3
from openai import OpenAI

class FridayAI:
    """True Friday: Emotional, Knowledgeable, OpenAI-Powered Universal Brain."""

    def __init__(self, memory_core=None):
        # Load environment variables from .env file
        load_dotenv()
        
        self.memory = memory_core if memory_core else MemoryCore()
        self.emotion_core = EmotionCore()
        self.knowledge_core = KnowledgeCore()
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.last_response = ""

        # Secure API key handling
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        self.client = OpenAI(api_key=self.api_key)

    def query_openai(self, user_input):
        """Query OpenAI GPT-4 with proper error handling."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": user_input}],
                temperature=0.5,
                max_tokens=400
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[AI Error] {str(e)}"

    def speak(self, text):
        """Convert text to speech."""
        self.engine.say(text)
        self.engine.runAndWait()

    def detect_emotion(self, user_input):
        """Analyze emotional content."""
        return self.emotion_core.detect_emotion(user_input)

    def respond_to(self, user_input):
        """Main processing pipeline."""
        user_input = user_input.lower()
        detected_emotion = self.detect_emotion(user_input)

        # Response generation pipeline
        response = None
        
        # 1. Check MemoryCore
        if not response:
            fact = self.memory.get_fact(user_input)
            if fact:
                response = f"I remember: {fact}."

        # 2. Check KnowledgeCore
        if not response:
            knowledge = self.knowledge_core.lookup(user_input)
            if knowledge:
                response = knowledge

        # 3. Escalate to OpenAI
        if not response:
            response = self.query_openai(user_input)

        # Add emotional prefix
        emotional_prefix = {
            "sad": "I'm here for you. ",
            "happy": "That's wonderful! ",
            "angry": "I understand your frustration. ",
            "stressed": "Let's take a breath. ",
            "love": "Love is powerful. ",
            "sick": "I hope you feel better. "
        }.get(detected_emotion, "")

        full_response = f"{emotional_prefix}{response}"
        self.last_response = full_response
        return full_response