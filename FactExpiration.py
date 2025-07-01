# FactExpiration.py

import time

class FactExpiration:
    def __init__(self, expiration_time_seconds=900):  # Default: 15 minutes
        self.fact_timestamps = {}
        self.expiration_time = expiration_time_seconds

    def update_fact(self, key):
        self.fact_timestamps[key] = time.time()

    def is_fact_expired(self, key):
        if key not in self.fact_timestamps:
            return True
        return (time.time() - self.fact_timestamps[key]) > self.expiration_time

    def remove_expired_facts(self, facts):
        to_remove = [k for k in facts if self.is_fact_expired(k)]
        for key in to_remove:
            facts.pop(key, None)
            self.fact_timestamps.pop(key, None)
        return facts
