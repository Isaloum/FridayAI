# =====================================
# MemoryCore.py - Complete Implementation with Empathy Flagging Support
# =====================================
import os
import json
import logging
import hashlib
import difflib
import re
from datetime import datetime, timedelta
from difflib import get_close_matches
from cryptography.fernet import Fernet
from collections import deque, Counter
from typing import Dict, List, Optional

class MemoryCore:
    """Advanced memory management system with version control, encryption, and contextual awareness"""

    def __init__(self, 
                 memory_file: str = "friday_memory.enc",
                 key_file: str = "memory.key",
                 max_context: int = 7,
                 conflict_threshold: float = 0.8):
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
        new_key = Fernet.generate_key()
        with open(self.key_file, 'wb') as f:
            f.write(new_key)
        return Fernet(new_key)

    def _init_logging(self) -> logging.Logger:
        """Configure memory operation logging"""
        logger = logging.getLogger("FridayAI.MemoryCore")
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler('memory_operations.log')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
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
        """Store information with version control and conflict detection"""
        norm_key = self._normalize_key(key)
        timestamp = datetime.now().isoformat()
        entry = {
            'value': value,
            'timestamp': timestamp,
            'source': source,
            'metadata': metadata or {}
        }

        if norm_key in self.memory:
            previous = self.memory[norm_key][-1]
            entry['previous_version'] = previous['timestamp']
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

    def flag_important_memory(self, memory_data: dict):
        """
        Flags emotionally significant entries (used by EmpathyReasoner).
        Stores them with 'flagged_' prefix for retrieval and review.
        """
        key = f"flagged_{datetime.now().isoformat()}"
        entry = {
            'value': memory_data.get("original_text", ""),
            'timestamp': memory_data.get("timestamp", datetime.now().isoformat()),
            'type': memory_data.get("type", "empathy_trigger"),
            'tags': memory_data.get("subtext_flags", []),
            'suggested_response': memory_data.get("suggested_response", None),
            'source': "EmpathyReasoner"
        }
        self.memory[key] = [entry]
        self.context_buffer.append(key)
        self.save_memory()
        self.logger.info(f"Flagged important memory: {key}")

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
        """Retrieve information with context-aware fallback"""
        norm_key = self._normalize_key(key)
        if norm_key in self.memory:
            return self._get_version(norm_key, version)
        if context_search:
            return self._context_fallback(key)
        return None
        
    def query(self, prompt: str, top_k: int = 3) -> list:
        """
        Respond with top_k relevant memory matches to the prompt.
        For now, it returns a single match repeated if available.
        """
        fact = self.get_fact(prompt)
        if fact:
            return [{"text": fact.get("value", "")}] * top_k
        return []

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
        candidates = get_close_matches(key, self.memory.keys(), n=3, cutoff=0.6)
        context_matches = [k for k in self.context_buffer if k in self.memory]
        all_candidates = list(set(candidates + context_matches))
        if not all_candidates:
            return None
        best_match = max(all_candidates, key=lambda k: self._calculate_context_score(k))
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

    def summarize_user_data(self, days: int = 7) -> str:
        """
        Summarizes memory contents from the past `days`.
        Returns top keywords, dominant emotions, and entry stats.
        """
        cutoff = datetime.now() - timedelta(days=days)
        recent_entries = []

        for key, versions in self.memory.items():
            for entry in versions:
                entry_time = datetime.fromisoformat(entry.get('timestamp', datetime.now().isoformat()))
                if entry_time >= cutoff:
                    recent_entries.append(entry)

        if not recent_entries:
            return f"No memory entries found in the last {days} days."

        all_text = " ".join([entry.get("value", "") if isinstance(entry.get("value"), str) else str(entry.get("value")) for entry in recent_entries])
        all_emotions = Counter()
        for entry in recent_entries:
            emotion = entry.get("metadata", {}).get("emotion", {})
            if isinstance(emotion, dict):
                all_emotions.update(emotion)

        words = re.findall(r'\b\w+\b', all_text.lower())
        stop_words = set(["the", "and", "you", "for", "with", "that", "this", "have", "are", "but", "was", "what", "can"])
        filtered_words = [w for w in words if w not in stop_words and len(w) > 3]
        keywords = Counter(filtered_words).most_common(10)

        summary = f"\U0001f9e0 Memory Summary for the Last {days} Days\n\n"
        summary += "\U0001f511 Top Topics:\n"
        summary += "\n".join([f"- {kw[0]} (×{kw[1]})" for kw in keywords]) or "None"
        summary += "\n\n\U0001f636‍\U0001f32b️ Dominant Emotions:\n"
        summary += "\n".join([f"- {em} (×{count})" for em, count in all_emotions.most_common()]) or "None"
        summary += f"\n\n📦 Entries Analyzed: {len(recent_entries)}"
        return summary

    def get_all(self) -> List[Dict]:
        """
        Returns a flat list of all memory entries across all keys and versions.
        Used for summarization, emotional reflection, and query filtering.
        """
        all_entries = []
        for versions in self.memory.values():
            all_entries.extend(versions)
        return all_entries

    def fetch_recent(self, limit: int = 10, within_minutes: int = 1440) -> List[Dict]:
        """
        Return the most recent `limit` entries within the past `within_minutes`.
        """
        cutoff = datetime.now() - timedelta(minutes=within_minutes)
        all_entries = self.get_all()
        filtered = [entry for entry in all_entries if datetime.fromisoformat(entry['timestamp']) >= cutoff]
        sorted_entries = sorted(filtered, key=lambda e: e['timestamp'], reverse=True)
        return sorted_entries[:limit]

    def inject_tasks_from_planner(self, task_file="core/knowledge_data/plans/task_queue.json"):
        """Load task queue and inject them as memory entries"""
        if not Path(task_file).exists():
            return "No tasks to inject."

        with open(task_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        timestamp = data.get("timestamp")
        tasks = data.get("tasks", [])
        
        for task in tasks:
            self.save_fact(
                key=f"task_{task}",
                value=task,
                source="planner_engine",
                metadata={"timestamp": timestamp, "type": "task"}
            )

        return f"Injected {len(tasks)} task(s) from planner."
