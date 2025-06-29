# =====================================
# Friday AI Assistant - Enhanced Version
# =====================================
import json
import os
import re
import random
from datetime import datetime
from difflib import get_close_matches
from fuzzywuzzy import fuzz
import pyttsx3
import speech_recognition as sr
from openai import OpenAI
from cryptography.fernet import Fernet

# =====================================
# Memory Core with Enhanced Features
# =====================================
class EnhancedMemoryCore:
    def __init__(self, memory_file='friday_personal_memory.json', key_file='memory.key'):
        self.memory_file = memory_file
        self.key_file = key_file
        self.short_term_memory = []
        self.cipher = self.init_cipher()
        self.memory = self.load_memory()

    def init_cipher(self):
        """Initialize encryption cipher"""
        from cryptography.fernet import Fernet
        if os.path.exists(self.key_file):
            with open(self.key_file, 'rb') as f:
                key = f.read()
        else:
            key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(key)
        return Fernet(key)

    def load_memory(self):
        """Load encrypted memory from file"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'rb') as f:
                    encrypted = f.read()
                    decrypted = self.cipher.decrypt(encrypted)
                    return json.loads(decrypted.decode())
            except Exception as e:
                print(f"[Memory Load Error] {e}")
                return {}
        return {}

    def save_memory(self):
        """Save memory with encryption"""
        try:
            encrypted = self.cipher.encrypt(
                json.dumps(self.memory).encode())
            with open(self.memory_file, 'wb') as f:
                f.write(encrypted)
        except Exception as e:
            print(f"[Memory Save Error] {e}")

    def add_context(self, text):
        """Add conversation context"""
        self.short_term_memory.append(text)
        if len(self.short_term_memory) > 5:
            self.short_term_memory.pop(0)

    def get_context(self):
        """Get recent conversation context"""
        return " ".join(self.short_term_memory)

    def save_fact(self, fact_key, fact_value, category="general"):
        """Save a fact with category and tracking"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.memory[fact_key.lower()] = {
            "value": fact_value,
            "category": category,
            "updated_at": timestamp,
            "access_count": 0
        }
        self.save_memory()
        return f"✅ Fact saved: {fact_key} = {fact_value}."

    def get_fact(self, fact_key):
        """Retrieve a fact with access tracking"""
        fact = self.memory.get(fact_key.lower())
        if fact:
            self.memory[fact_key.lower()]["access_count"] += 1
            self.save_memory()
            return fact["value"]
        return None

    def find_closest_fact(self, query, threshold=75):
        """Fuzzy matching for fact retrieval"""
        best_match, highest_score = None, 0
        for fact in self.memory.keys():
            score = fuzz.token_set_ratio(query, fact)
            if score > highest_score and score >= threshold:
                highest_score, best_match = score, fact
        return best_match

    def list_facts(self):
        """List all facts with metadata"""
        if not self.memory:
            return "No facts stored yet."
        return "\n".join(
            f"{k}: {v['value']} (Category: {v['category']}, Accessed: {v['access_count']}x)"
            for k, v in self.memory.items()
        )

# =====================================
# Natural Language Understanding
# =====================================
class NLUProcessor:
    def __init__(self, memory_core):
        self.memory_core = memory_core
        self.command_handlers = {
            'remember': self.handle_remember,
            'update': self.handle_update,
            'delete': self.handle_delete,
            'what is my': self.handle_query,
            'list': self.handle_list,
            'what can you do': self.handle_skills
        }

    def parse_intent(self, user_input):
        """Determine user intent with fallback handling"""
        lower_input = user_input.lower()
        for prefix, handler in self.command_handlers.items():
            if lower_input.startswith(prefix):
                try:
                    return handler(user_input)
                except Exception as e:
                    return self.get_fallback_response(prefix)
        return None

    def handle_remember(self, text):
        """Process remember commands"""
        pattern = r"remember (my |that )?(.+?) (is|are|was|were) (.+)"
        match = re.match(pattern, text, re.IGNORECASE)
        if not match:
            raise ValueError("Invalid remember format")
        return self.memory_core.save_fact(match.group(2).strip(), match.group(4).strip())

    def handle_query(self, text):
        """Process fact queries"""
        fact_key = text.lower().split("what is my")[1].strip().replace("?", "")
        fact = self.memory_core.get_fact(fact_key) or \
               self.memory_core.get_fact(self.memory_core.find_closest_fact(fact_key) or "")
        return f"Your {fact_key} is {fact}" if fact else f"I don't know your {fact_key}"

    def handle_list(self, text):
        """Process list requests"""
        return self.memory_core.list_facts()

    def get_fallback_response(self, command_type):
        """User-friendly error messages"""
        fallbacks = {
            'remember': "Try: 'Remember that my favorite color is blue'",
            'update': "Format: 'Update [fact] to [new value]'",
            'delete': "Format: 'Delete [fact name]'"
        }
        return fallbacks.get(command_type, "I didn't understand that command.")

# =====================================
# Personality Engine
# =====================================
class PersonalityEngine:
    def __init__(self):
        self.traits = {
            "formality": 0.5,
            "humor": 0.3,
            "empathy": 0.7
        }
        self.energy_level = 100
        self.mood = "neutral"

    def update_state(self, interaction_type):
        """Update energy and mood based on interaction"""
        energy_impact = {
            'simple_query': -2,
            'complex_task': -10,
            'positive_feedback': +15,
            'negative_feedback': -20
        }
        self.energy_level = max(0, min(100, 
            self.energy_level + energy_impact.get(interaction_type, 0)))

    def infuse_personality(self, text):
        """Modify response based on personality traits"""
        # Energy impact
        if self.energy_level < 30:
            text = f"⚡[Low Energy] {text.lower()}"
        elif self.energy_level > 80:
            text = f"{text.upper()}!! 😃"
        
        # Humor injection
        if random.random() < self.traits["humor"]:
            jokes = [" *beep boop*", " That's what she said!", " 🤖"]
            text += random.choice(jokes)
            
        return text

# =====================================
# Skill Management System
# =====================================
class SkillManager:
    def __init__(self):
        self.skills = {
            'weather': self.WeatherSkill(),
            'timer': self.TimerSkill()
        }

    class WeatherSkill:
        def __init__(self):
            self.description = "Get weather forecasts"
            
        def execute(self, params):
            location = params.get('location', 'unknown location')
            return f"Weather in {location}: 72°F and sunny (sample response)"

    class TimerSkill:
        def __init__(self):
            self.description = "Set timers and reminders"
            
        def execute(self, params):
            duration = params.get('duration', '5 minutes')
            return f"Timer set for {duration} ⏱️"

    def execute_skill(self, skill_name, params):
        """Execute a skill if available"""
        skill = self.skills.get(skill_name.lower())
        return skill.execute(params) if skill else None

    def list_skills(self):
        """List available skills"""
        return "\n".join(f"• {name}: {skill.description}" 
                        for name, skill in self.skills.items())

# =====================================
# Main Friday AI Class
# =====================================
class FridayAI:
    def __init__(self):
        self.memory_core = EnhancedMemoryCore()
        self.nlu = NLUProcessor(self.memory_core)
        self.personality = PersonalityEngine()
        self.skills = SkillManager()
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.client = OpenAI(api_key="sk-proj-Ax6Hn09TqRSjpXdZ18weyZsQ1uI4YdJ2nIDbNfczPHXblF8VzmVIQoZftHW2C8CUKAIROsJ1YET3BlbkFJO6COwiUwzQq6hyRUog4y7DEtI7srCNP1QVnZrO6VTcFg-MrKaopniMGbU39Yh_jNCFTz3MTbgAhi")  # Replace with your key

    def speak(self, text):
        """Convert text to speech"""
        self.engine.say(text)
        self.engine.runAndWait()

    def listen(self):
        """Listen for voice input"""
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("[Listening...]")
            audio = recognizer.listen(source)
        try:
            return recognizer.recognize_google(audio)
        except Exception:
            return ""

    def chat_with_openai(self, prompt):
        """Query GPT-3.5 with context"""
        try:
            context = self.memory_core.get_context()
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"Context: {context}\nYou are Friday, a helpful AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"

    def process_input(self, user_input):
        """Main processing pipeline"""
        self.memory_core.add_context(f"User: {user_input}")
        
        # Try NLU first
        response = self.nlu.parse_intent(user_input)
        
        # Then try skills
        if not response and "set a timer for" in user_input.lower():
            response = self.skills.execute_skill('timer', 
                {'duration': user_input.lower().split("for")[1].strip()})
        
        # Fallback to GPT
        if not response:
            response = self.chat_with_openai(user_input)
            
        # Add personality and context
        response = self.personality.infuse_personality(response)
        self.memory_core.add_context(f"Friday: {response}")
        return response

    def run(self):
        """Main interaction loop"""
        print("Friday AI Assistant activated. How can I help?")
        while True:
            # Listen for wake word
            user_input = self.listen()
            if "friday" in user_input.lower():
                self.speak("Yes?")
                
                # Conversation loop
                while True:
                    command = self.listen()
                    if not command:
                        continue
                    if "goodbye" in command.lower():
                        self.speak("Goodbye!")
                        return
                        
                    response = self.process_input(command)
                    print(f"Friday: {response}")
                    self.speak(response)

# =====================================
# Run the Assistant
# =====================================
if __name__ == "__main__":
    assistant = FridayAI()
    assistant.run()