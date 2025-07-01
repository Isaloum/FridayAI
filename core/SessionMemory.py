# =====================================
# SessionMemory.py – Tracks temporal continuity & last context
# =====================================

from datetime import datetime, timedelta

class SessionMemory:
    def __init__(self):
        self.session_log = []
        self.last_user_input = ""
        self.last_emotion = "neutral"
        self.last_timestamp = datetime.now()
        self.state = {}  # For key-value flags like 'already_greeted', 'last_user'

    def update(self, user_input: str, emotion: str):
        timestamp = datetime.now()
        self.last_user_input = user_input
        self.last_emotion = emotion
        self.last_timestamp = timestamp
        self.session_log.append({
            "timestamp": timestamp.isoformat(),
            "input": user_input,
            "emotion": emotion
        })

    def get_context_summary(self, limit: int = 3) -> str:
        recent = self.session_log[-limit:]
        return "\n".join([
            f"[{entry['timestamp'].split('T')[0]}] {entry['emotion'].capitalize()}: {entry['input']}"
            for entry in recent
        ])

    def get_tone_shift(self) -> str:
        now = datetime.now()
        gap = now - self.last_timestamp

        if gap > timedelta(minutes=3):
            return "reflective"
        return "normal"

    def reset(self):
        self.session_log.clear()
        self.last_user_input = ""
        self.last_emotion = "neutral"
        self.last_timestamp = datetime.now()
        self.state.clear()

    # === Internal key-value memory API ===
    def load(self, key: str):
        return self.state.get(key)

    def save(self, key: str, value):
        self.state[key] = value

    def get(self, key: str, default=None):
        result = self.load(key)
        return result if result is not None else default

    def set(self, key: str, value):
        self.save(key, value)
