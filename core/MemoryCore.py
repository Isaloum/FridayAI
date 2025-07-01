# =============================================
# File: core/MemoryCore.py  
# Purpose: JSON-backed memory manager for saving and querying cognitive data
# Location: Must be in core/ folder
# =============================================

import json
from pathlib import Path
from datetime import datetime, timedelta
import os
#print("[DEBUG] MemoryCore loaded from:", os.path.abspath(__file__))

MEMORY_PATH = Path("core/memory_bank/memory_log.json")

class MemoryCore:
    def __init__(self, memory_file="friday_memory.enc", key_file="memory.key"):
        self.memory_file = memory_file
        self.key_file = key_file
        # Fix: Add the missing path property
        self.path = MEMORY_PATH
        #print(f"[DEBUG] MemoryCore loaded from: {__file__}")
        
        # Create memory directory and file if they don't exist
        self._ensure_memory_file_exists()

    def _ensure_memory_file_exists(self):
        """
        Make sure the memory file and directory exist.
        Create empty file if needed.
        """
        # Create directory if it doesn't exist
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create empty memory file if it doesn't exist
        if not self.path.exists():
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump([], f)  # Start with empty list
            print(f"[DEBUG] Created new memory file: {self.path}")

    def get_all(self):
        """
        Get all memory entries from the file.
        This method was missing and causing the error.
        """
        if not self.path.exists():
            return []
            
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception as e:
            print(f"[ERROR] Failed to read memory file: {e}")
            return []

    def save_memory(self, memory):
        """
        Save a new memory entry to the file.
        """
        print("[DEBUG] Using updated MemoryCore.save_memory")
        print("[DEBUG] Saving memory entry:")
        print(json.dumps(memory, indent=2))
        
        # Add timestamp if not present
        memory["timestamp"] = memory.get("timestamp", datetime.now().isoformat())
        
        # Ensure file exists
        self._ensure_memory_file_exists()
        
        try:
            with open(self.path, 'r+', encoding='utf-8') as f:
                data = json.load(f)
                data.append(memory)
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()
        except Exception as e:
            print(f"[ERROR] Failed to save memory: {e}")

    def query_recent(self, limit=10):
        """
        Load recent memory entries from disk up to the given limit.
        Entries are assumed to be stored chronologically in the JSON file.
        """
        if not self.path.exists():
            return []

        with open(self.path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data[-limit:] if len(data) >= limit else data

    def query_memories(self, filter_tags=None, since_hours=24):
        """
        Query memories with optional filtering by tags and time.
        """
        if not self.path.exists():
            return []

        with open(self.path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        recent = datetime.now() - timedelta(hours=since_hours)
        results = []
        for entry in data:
            ts = datetime.fromisoformat(entry.get("timestamp", datetime.now().isoformat()))
            if ts >= recent:
                if filter_tags is None or any(tag in entry.get("tags", []) for tag in filter_tags):
                    results.append(entry)

        return results
        
    def get_recent_entries(self, entry_type: str = None, days: int = 7, limit: int = 10):
        """
        Retrieve entries from memory matching a specific type within the last `days`.
        """
        if not self.path.exists():
            return []

        with open(self.path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        threshold = datetime.now() - timedelta(days=days)
        results = []

        for entry in data:
            try:
                timestamp = datetime.fromisoformat(entry.get("timestamp", "1970-01-01T00:00:00"))
            except ValueError:
                continue  # skip entries with invalid timestamps

            if entry_type and entry.get("type") != entry_type:
                continue

            if timestamp >= threshold:
                results.append(entry)

        return results

    def get_memory(self, filter=None):
        """
        Compatibility wrapper for legacy interfaces expecting 'get_memory()'.
        Uses 'get_recent_entries()' to fetch recent logs of a given type.
        """
        entry_type = filter.get("type") if filter else None
        return self.get_recent_entries(entry_type=entry_type, days=7)

    def get_recent_emotions(self, limit=3):
        """
        Get recent emotional entries from memory.
        """
        recent = self.get_recent_entries(entry_type="pregnancy_log", days=7)
        return [e for e in recent if e.get("emotion")] [-limit:]