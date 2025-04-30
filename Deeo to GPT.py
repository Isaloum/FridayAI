import json
import os
from datetime import datetime
from cryptography.fernet import Fernet
import hashlib

class MemoryCore:
    """Secure memory storage with case-insensitive access and validation"""
    
    def __init__(self, memory_file='memory.json', key_file='memory.key'):
        self.memory_file = memory_file
        self.key_file = key_file
        self.cipher = self._init_cipher()
        self.memory = self._load_memory()
        self.backup_file = f"{memory_file}.bak"

    def _init_cipher(self):
        """Initialize encryption with key rotation support"""
        if os.path.exists(self.key_file):
            with open(self.key_file, 'rb') as f:
                return Fernet(f.read())
        key = Fernet.generate_key()
        with open(self.key_file, 'wb') as f:
            f.write(key)
        return Fernet(key)

    def _load_memory(self):
        """Load encrypted memory with corruption handling"""
        if not os.path.exists(self.memory_file):
            return {}

        try:
            with open(self.memory_file, 'rb') as f:
                encrypted = f.read()
                decrypted = self.cipher.decrypt(encrypted)
                return json.loads(decrypted.decode())
        except (json.JSONDecodeError, Fernet.InvalidToken):
            return self._restore_backup()

    def _save_memory(self):
        """Atomic encrypted save with backup"""
        try:
            # Write to temporary file first
            temp_file = f"{self.memory_file}.tmp"
            encrypted = self.cipher.encrypt(json.dumps(self.memory).encode())
            
            with open(temp_file, 'wb') as f:
                f.write(encrypted)
            
            # Replace original after successful write
            os.replace(temp_file, self.memory_file)
            self._create_backup()
            
        except Exception as e:
            print(f"Save failed: {str(e)}")
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def _create_backup(self):
        """Create versioned backup"""
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'rb') as src, open(self.backup_file, 'wb') as dst:
                dst.write(src.read())

    def _restore_backup(self):
        """Restore from backup if available"""
        if os.path.exists(self.backup_file):
            with open(self.backup_file, 'rb') as f:
                encrypted = f.read()
                decrypted = self.cipher.decrypt(encrypted)
                return json.loads(decrypted.decode())
        return {}

    def _hash_key(self, key):
        """Case-insensitive hashing with original key preservation"""
        return hashlib.sha256(key.lower().encode()).hexdigest()

    def save_fact(self, fact_key, fact_value):
        """Store fact with validation and case preservation"""
        if not self.validate_fact(fact_key, fact_value):
            raise ValueError("Invalid fact key/value combination")
            
        hashed_key = self._hash_key(fact_key)
        self.memory[hashed_key] = {
            'original_key': fact_key,
            'value': fact_value,
            'timestamp': datetime.now().isoformat()
        }
        self._save_memory()

    def update_fact(self, fact_key, new_value):
        """Update existing fact with validation"""
        hashed_key = self._hash_key(fact_key)
        if hashed_key not in self.memory:
            raise KeyError(f"Fact '{fact_key}' not found")
            
        if not self.validate_fact(fact_key, new_value):
            raise ValueError("Invalid value for fact update")
            
        self.memory[hashed_key]['value'] = new_value
        self.memory[hashed_key]['timestamp'] = datetime.now().isoformat()
        self._save_memory()

    def get_fact(self, fact_key):
        """Retrieve fact with original casing"""
        hashed_key = self._hash_key(fact_key)
        entry = self.memory.get(hashed_key)
        return entry['value'] if entry else None

    def validate_fact(self, fact_key, fact_value):
        """Enhanced validation with security checks"""
        # Prevent injection attacks
        if any(c in fact_key for c in [';', '<', '>', '\\']):
            return False
            
        # Validate value length and type
        if len(fact_key) > 100 or len(str(fact_value)) > 1000:
            return False
            
        # Allow any keys but filter dangerous patterns
        forbidden_patterns = ['password', 'secret', 'token']
        if any(pattern in fact_key.lower() for pattern in forbidden_patterns):
            return False
            
        return True