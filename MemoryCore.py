# =====================================
# MemoryCore.py
# Friday True Memory Core
# =====================================

import json
import os
import logging
from datetime import datetime
from difflib import get_close_matches
from cryptography.fernet import Fernet
from collections import deque

class MemoryCore:
    """Handles secure memory management, fuzzy matching, and context tracking."""

    def __init__(self, memory_file='friday_personal_memory.json', key_file='memory.key'):
        self.memory_file = memory_file
        self.key_file = key_file
        self.logger = logging.getLogger("FridayAI.MemoryCore")
        self.cipher = self.init_cipher()
        self.memory = self.load_memory()
        self.context_memory = deque(maxlen=10)  # Fixed-size, auto-trimmed context buffer

    def init_cipher(self):
        """Initialize or load encryption cipher."""
        if os.path.exists(self.key_file):
            with open(self.key_file, 'rb') as f:
                key = f.read()
        else:
            key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(key)
        return Fernet(key)

    def load_memory(self):
        """Load encrypted memory from file."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'rb') as f:
                    encrypted = f.read()
                    decrypted = self.cipher.decrypt(encrypted)
                    return json.loads(decrypted.decode())
            except Exception as e:
                self.logger.error(f"[Memory Load Error] {e}")
                return {}
        return {}

    def save_memory(self):
        """Encrypt and persist memory."""
        try:
            encrypted = self.cipher.encrypt(json.dumps(self.memory).encode())
            with open(self.memory_file, 'wb') as f:
                f.write(encrypted)
        except Exception as e:
            self.logger.error(f"[Memory Save Error] {e}")

    def _normalize_key(self, key):
        """Normalize memory keys to ensure consistent access."""
        return key.lower().replace("'", "").replace(" ", "_")

    def save_fact(self, fact_key, fact_value):
        """Save or update a memory fact."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        key = self._normalize_key(fact_key)
        self.memory[key] = {    
            "value": fact_value,
            "updated_at": timestamp,
            "access_count": 0
        }
        self.save_memory()
        return f"✅ Saved: {key} = {fact_value}"

    def add_fact(self, key, value):
        """Alias for save_fact for external use."""
        return self.save_fact(key, value)

    def get_fact(self, key):
        """Retrieve a fact, with fuzzy fallback if exact not found."""
        normalized_key = self._normalize_key(key)
        fact = self.memory.get(normalized_key)

        used_key = normalized_key
        if not fact:
            closest = self.find_closest_fact(normalized_key)
            if closest:
                used_key = closest
                fact = self.memory.get(closest)

        if fact:
            fact["access_count"] += 1
            fact["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.save_memory()
            return {"key": used_key, "value": fact["value"]}

        return None

