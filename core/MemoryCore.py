# =====================================
# MemoryCore.py - Complete Implementation
# =====================================
import os
import json
import logging
import hashlib
from datetime import datetime
from difflib import get_close_matches
from cryptography.fernet import Fernet
from collections import deque
from typing import Dict, List, Optional

class MemoryCore:
    """Advanced memory management system with version control, encryption, and contextual awareness"""
    
    def __init__(self, 
                 memory_file: str = "friday_memory.enc",
                 key_file: str = "memory.key",
                 max_context: int = 7,
                 conflict_threshold: float = 0.8):
        """
        Initialize secure memory system
        
        :param memory_file: Encrypted memory storage file
        :param key_file: Encryption key file
        :param max_context: Size of recent context buffer
        :param conflict_threshold: Similarity threshold for conflicts
        """
        self.memory_file = memory_file
        self.key_file = key_file
        self.conflict_threshold = conflict_threshold
        self.context_buffer = deque(maxlen=max_context)
        self.operation_log = []
        
        # Initialize subsystems
        self.cipher = self._init_cipher_system()
        self.memory = self._load_memory()
        self.logger = self._init_logging()

    def _init_cipher_system(self) -> Fernet:
        """Initialize encryption infrastructure"""
        if os.path.exists(self.key_file):
            with open(self.key_file, 'rb') as f:
                return Fernet(f.read())
        
        # Generate new key if none exists
        new_key = Fernet.generate_key()
        with open(self.key_file, 'wb') as f:
            f.write(new_key)
        return Fernet(new_key)

    def _init_logging(self) -> logging.Logger:
        """Configure memory operation logging"""
        logger = logging.getLogger("FridayAI.MemoryCore")
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler('memory_operations.log')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(handler)
        return logger

    def _load_memory(self) -> Dict:
        """Load and decrypt memory storage"""
        if not os.path.exists(self.memory_file):
            return {}
            
        try:
            with open(self.memory_file, 'rb') as f:
                encrypted_data = f.read()
                decrypted = self.cipher.decrypt(encrypted_data)
                return json.loads(decrypted.decode())
        except Exception as e:
            self.logger.error(f"Memory load failed: {str(e)}")
            return {}

    def save_memory(self) -> None:
        """Encrypt and persist memory state"""
        try:
            encrypted = self.cipher.encrypt(json.dumps(self.memory).encode())
            with open(self.memory_file, 'wb') as f:
                f.write(encrypted)
        except Exception as e:
            self.logger.error(f"Memory save failed: {str(e)}")

    def _normalize_key(self, key: str) -> str:
        """Standardize memory keys for consistent access"""
        return hashlib.sha256(key.lower().encode()).hexdigest()[:32]

    def save_fact(self, 
                 key: str, 
                 value: object, 
                 source: str = "user",
                 metadata: Optional[Dict] = None) -> Dict:
        """
        Store information with version control and conflict detection
        
        :param key: Fact identifier
        :param value: Information to store
        :param source: Information source (system|user|auto_learned)
        :param metadata: Additional context about the fact
        :return: Operation result
        """
        norm_key = self._normalize_key(key)
        timestamp = datetime.now().isoformat()
        entry = {
            'value': value,
            'timestamp': timestamp,
            'source': source,
            'metadata': metadata or {}
        }

        # Version control and conflict detection
        if norm_key in self.memory:
            previous = self.memory[norm_key][-1]
            entry['previous_version'] = previous['timestamp']
            
            # Detect conflicts
            if self._detect_conflict(previous['value'], value):
                conflict_entry = {
                    'key': norm_key,
                    'old_value': previous['value'],
                    'new_value': value,
                    'timestamp': timestamp,
                    'resolved': False
                }
                self.operation_log.append(('conflict', conflict_entry))
                self.logger.warning(f"Memory conflict detected: {norm_key}")

            self.memory[norm_key].append(entry)
        else:
            self.memory[norm_key] = [entry]

        self.context_buffer.append(norm_key)
        self.save_memory()
        return {'status': 'success', 'key': norm_key, 'version': len(self.memory[norm_key])}

    def _detect_conflict(self, old_val, new_val) -> bool:
        """Determine if new value conflicts with existing knowledge"""
        if isinstance(old_val, str) and isinstance(new_val, str):
            similarity = self._calculate_similarity(old_val, new_val)
            return similarity < self.conflict_threshold
        return old_val != new_val

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Compute string similarity score"""
        seq = difflib.SequenceMatcher(None, str1.lower(), str2.lower())
        return seq.ratio()

    def get_fact(self, 
                key: str, 
                version: int = -1,
                context_search: bool = True) -> Optional[Dict]:
        """
        Retrieve information with context-aware fallback
        
        :param key: Fact identifier to retrieve
        :param version: Specific version to retrieve (-1 for latest)
        :param context_search: Enable contextual fuzzy matching
        :return: Requested fact or None
        """
        norm_key = self._normalize_key(key)
        
        # Direct match
        if norm_key in self.memory:
            return self._get_version(norm_key, version)

        # Contextual fuzzy search
        if context_search:
            return self._context_fallback(key)
            
        return None

    def _get_version(self, norm_key: str, version: int) -> Optional[Dict]:
        """Retrieve specific version of a fact"""
        try:
            if version < 0:
                return self.memory[norm_key][version]
            return self.memory[norm_key][version]
        except (IndexError, KeyError):
            return None

    def _context_fallback(self, key: str) -> Optional[Dict]:
        """Context-aware fuzzy search implementation"""
        # Combine exact and contextual matches
        candidates = get_close_matches(key, self.memory.keys(), n=3, cutoff=0.6)
        context_matches = [k for k in self.context_buffer if k in self.memory]
        
        # Prioritize contextually recent matches
        all_candidates = list(set(candidates + context_matches))
        
        if not all_candidates:
            return None
            
        # Select most contextually relevant
        best_match = max(
            all_candidates,
            key=lambda k: self._calculate_context_score(k)
        )
        
        self.context_buffer.append(best_match)
        return self.memory[best_match][-1]

    def _calculate_context_score(self, key: str) -> float:
        """Determine contextual relevance score"""
        base_score = self.context_buffer.count(key) / len(self.context_buffer)
        recency_bonus = 0.5 * (1 - (len(self.context_buffer) - list(self.context_buffer).index(key)) / len(self.context_buffer))
        return base_score + recency_bonus

    def bulk_learn(self, facts: Dict[str, object], source: str) -> Dict:
        """Batch process multiple facts"""
        results = {}
        for key, value in facts.items():
            result = self.save_fact(key, value, source)
            results[key] = result.get('version', 0)
        return {'processed': len(facts), 'results': results}

    def memory_report(self) -> Dict:
        """Generate system health diagnostics"""
        return {
            'total_facts': len(self.memory),
            'total_versions': sum(len(v) for v in self.memory.values()),
            'active_context': list(self.context_buffer),
            'unresolved_conflicts': sum(1 for e in self.operation_log if e[0] == 'conflict' and not e[1].get('resolved')),
            'storage_size': os.path.getsize(self.memory_file),
            'last_operation': self.operation_log[-1][0] if self.operation_log else None
        }

    def resolve_conflict(self, key: str, keep_version: int) -> bool:
        """Manually resolve memory conflicts"""
        norm_key = self._normalize_key(key)
        if norm_key not in self.memory:
            return False
            
        try:
            self.memory[norm_key] = [self.memory[norm_key][keep_version]]
            for entry in self.operation_log:
                if entry[0] == 'conflict' and entry[1]['key'] == norm_key:
                    entry[1]['resolved'] = True
            return True
        except IndexError:
            return False

# =====================
# Testing Harness
# =====================
if __name__ == "__main__":
    print("MemoryCore Validation Test")
    
    # Test configuration
    test_config = {
        'memory_file': 'test_memory.enc',
        'key_file': 'test.key'
    }
    
    # Initialize clean system
    if os.path.exists(test_config['memory_file']):
        os.remove(test_config['memory_file'])
    if os.path.exists(test_config['key_file']):
        os.remove(test_config['key_file'])
    
    mc = MemoryCore(**test_config)
    
    # Test basic operations
    mc.save_fact("user.name", "John Doe", "system")
    mc.save_fact("user.location", "New York", "user")
    
    # Test conflict
    mc.save_fact("user.location", "Los Angeles", "auto_learned")
    
    # Test retrieval
    print("Current location:", mc.get_fact("user.location"))
    print("Memory Report:", mc.memory_report())
    
    # Cleanup
    os.remove(test_config['memory_file'])
    os.remove(test_config['key_file'])