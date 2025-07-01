# NameToneLimiter.py
# --------------------------------------
# Controls when Friday should mention the user's name based on tone, time, and natural flow.

from datetime import datetime, timedelta

class NameToneLimiter:
    def __init__(self):
        self.last_used = None
        self.cooldown_minutes = 10  # avoid repeating name too often

    def should_use_name(self, emotion: str, tone_shift: str, is_first_message: bool = False) -> bool:
        """
        Determine whether Friday should use the user's name based on context.
        :param emotion: Detected emotional tone of the user
        :param tone_shift: Detected change in session tone (e.g., reconnect, gap, etc.)
        :param is_first_message: Boolean flag for first message of session
        :return: Boolean
        """
        now = datetime.now()

        # Always allow for first greeting or re-engagement
        if is_first_message or tone_shift in ["reconnect", "gap"]:
            self.last_used = now
            return True

        # Avoid repeating if recently used
        if self.last_used and (now - self.last_used) < timedelta(minutes=self.cooldown_minutes):
            return False

        # Use for assertive/emotional emphasis
        if emotion in ["urgent", "angry", "grateful", "worried"]:
            self.last_used = now
            return True

        return False
