# ==============================================
# File: core/SelfNarrativeCore.py
# Purpose: Tracks Friday's narrative, mood, and logs
# ==============================================

"""
SelfNarrativeCore.py
---------------------
Builds a persistent narrative of Friday's evolution.

- Logs key user inputs and AI replies
- Tracks mood changes over time
- Stores personality reflections

Usage:
    from core.SelfNarrativeCore import log_event, update_mood
"""

import os
import json
from datetime import datetime

LOG_PATH = "memory/self_narrative_log.json"
current_mood = "neutral"  # In-memory mood state

def _load_log():
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r") as f:
            return json.load(f)
    return []

def _save_log(data):
    with open(LOG_PATH, "w") as f:
        json.dump(data, f, indent=2)

def log_event(text: str, mood: str = None, source: str = "friday"):
    data = _load_log()
    entry = {
        "timestamp": datetime.now().isoformat(),
        "text": text,
        "source": source,
        "mood": mood or current_mood
    }
    data.append(entry)
    _save_log(data)

def update_mood(new_mood: str):
    global current_mood
    current_mood = new_mood
    
class SelfNarrativeCore:
    """
    Class version of SelfNarrativeCore for Friday's identity tracking.
    Handles personality and self-awareness evolution.
    """
    
    def __init__(self):
        """
        Initialize Friday's self-narrative system.
        """
       #print("[DEBUG] SelfNarrativeCore class initialized")
        
    def log_event(self, text: str, mood: str = None, source: str = "friday", kind: str = "event"):
        """
        Log an event to Friday's narrative.
        Uses the global log_event function.
        """
        log_event(text, mood, source)
        
    def update_mood(self, new_mood: str):
        """
        Update Friday's current mood.
        Uses the global update_mood function.
        """
        update_mood(new_mood)
