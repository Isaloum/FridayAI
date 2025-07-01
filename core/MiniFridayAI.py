# =====================================
# FridayAI - Consolidated Core System
# =====================================
import os
import re
import json
import logging
from datetime import datetime, timedelta
from difflib import get_close_matches
from typing import Dict, Optional, List
from collections import deque
from cryptography.fernet import Fernet
import pyttsx3
from dotenv import load_dotenv

# Install required packages:
# pip install pyttsx3 python-dotenv cryptography

# ========================
# MEMORY CORE IMPLEMENTATION
# ========================
class MemoryCore:
    """Advanced memory management with version control and encryption"""
    
    def __init__(self, memory_file='memory.json', key_file='memory.key'):
        self.memory_file = memory_file
        self.key_file = key_file
        self.cipher = self._init_cipher()
        self.memory = self._load_memory()
        self.context_stack = deque(maxlen=7)
        self.conflict_log = []

    def _init_cipher(self):
        """Initialize encryption system"""
        if os.path.exists(self.key_file):
            with open(self.key_file, 'rb') as f:
                return Fernet(f.read())
        key = Fernet.generate_key()
        with open(self.key_file, 'wb') as f:
            f.write(key)
        return Fernet(key)

    def _load_memory(self):
        """Load and decrypt memory"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'rb') as f:
                    return json.loads(self.cipher.decrypt(f.read()).decode())
            except:
                return {}
        return {}

    def save_memory(self):
        """Encrypt and save memory state"""
        with open(self.memory_file, 'wb') as f:
            f.write(self.cipher.encrypt(json.dumps(self.memory).encode()))

    def store(self, key: str, value: object, source: str = 'user') -> dict:
        """Version-controlled memory storage"""
        key = key.lower().replace(' ', '_')
        entry = {
            'value': value,
            'timestamp': datetime.now().isoformat(),
            'source': source,
            'versions': []
        }
        
        if key in self.memory:
            entry['versions'] = self.memory[key]['versions'] + [self.memory[key]['value']]
            self._detect_conflict(key, value)
            
        self.memory[key] = entry
        self.save_memory()
        return entry

    def _detect_conflict(self, key: str, new_value: str):
        """Conflict detection system"""
        old_value = self.memory[key]['value']
        if old_value != new_value:
            self.conflict_log.append({
                'key': key,
                'old': old_value,
                'new': new_value,
                'timestamp': datetime.now().isoformat()
            })

    def recall(self, key: str, version: int = -1) -> Optional[dict]:
        """Context-aware memory retrieval"""
        key = key.lower().replace(' ', '_')
        self.context_stack.append(key)
        
        if entry := self.memory.get(key):
            if version < 0 or version >= len(entry['versions']):
                return entry['value']
            return entry['versions'][version]
            
        # Fuzzy search with context weighting
        matches = get_close_matches(key, self.memory.keys(), n=3, cutoff=0.6)
        for match in matches:
            if match in self.context_stack:
                return self.memory[match]['value']
        return matches[0] if matches else None

# ======================
# AUTOLEARNING CORE
# ======================
class AutoLearningCore:
    """Pattern-based autonomous knowledge acquisition"""
    
    def __init__(self, memory: MemoryCore):
        self.memory = memory
        self.patterns = [
            (re.compile(r'my (name|identifier) (?:is|am) (.+)', re.I), 'identity/name'),
            (re.compile(r'(?:live in|located in|based in) (.+)', re.I), 'geo/location'),
            (re.compile(r'(?:born on|birthdate) (\d{4}-\d{2}-\d{2})', re.I), 'bio/birthdate'),
            (re.compile(r'work at|employed at (.+)', re.I), 'employment/company')
        ]

    def process(self, text: str) -> dict:
        """Main learning interface"""
        for pattern, key in self.patterns:
            if match := pattern.search(text):
                self.memory.store(key, match.group(1).strip(), 'auto_learned')
                return {'learned': True, 'key': key}
        return {'learned': False}

# =====================
# MAIN AI PROCESSOR
# =====================
class FridayAI:
    """Central cognitive processing unit"""
    
    def __init__(self):
        self.memory = MemoryCore()
        self.learner = AutoLearningCore(self.memory)
        self.voice = self._init_voice()
        self.log = self._init_logging()

    def _init_voice(self):
        """Initialize text-to-speech"""
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 160)
            return engine
        except:
            return None

    def _init_logging(self):
        """Set up activity tracking"""
        logger = logging.getLogger('FridayAI')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler('activity.log')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'
        ))
        logger.addHandler(handler)
        return logger

    def process(self, text: str) -> dict:
        """Main processing pipeline"""
        result = {'input': text, 'timestamp': datetime.now().isoformat()}
        
        # Learning phase
        learn_result = self.learner.process(text)
        
        # Response generation
        response = self._generate_response(text, learn_result)
        
        # Voice output
        if self.voice and response.get('speech'):
            self.voice.say(response['speech'])
            self.voice.runAndWait()
            
        result.update(response)
        return result

    def _generate_response(self, text: str, learning: dict) -> dict:
        """Response generation logic"""
        if learning['learned']:
            return {
                'response': f"Learned: {learning['key']}",
                'speech': "I'll remember that!",
                'type': 'learning'
            }
            
        # Add your custom response logic here
        return {
            'response': "Interesting, tell me more!",
            'speech': "Fascinating input, please continue.",
            'type': 'engagement'
        }

# ====================
# INTERFACE SYSTEM
# ====================
class FridayInterface:
    """User interaction handler"""
    
    def __init__(self):
        self.ai = FridayAI()
        
    def chat(self):
        """Start interactive chat session"""
        print("Friday AI System [v2.4]")
        print("Type 'exit' to end session\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ('exit', 'quit'):
                    break
                    
                response = self.ai.process(user_input)
                print(f"\nFriday: {response['response']}\n")
                
            except KeyboardInterrupt:
                print("\nSession terminated")
                break

# ================
# SYSTEM BOOTSTRAP
# ================
if __name__ == "__main__":
    interface = FridayInterface()
    interface.chat()