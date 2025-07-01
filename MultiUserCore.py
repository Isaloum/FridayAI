# MultiUserCore.py – User Profile Manager for FridayAI

import os
import json
from datetime import datetime

class MultiUserCore:
    def __init__(self, base_path="./memory/users"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def _user_dir(self, user_id: str):
        return os.path.join(self.base_path, user_id)

    def _user_file(self, user_id: str, filename: str):
        return os.path.join(self._user_dir(user_id), filename)

    def load_user_profile(self, user_id: str) -> dict:
        """Load all components: tone, identity, memory"""
        os.makedirs(self._user_dir(user_id), exist_ok=True)
        try:
            with open(self._user_file(user_id, "profile.json"), "r") as f:
                return json.load(f)
        except:
            # Default profile
            profile = {
                "personality": {
                    "warmth": 0.5,
                    "humor": 0.5,
                    "precision": 0.5,
                    "formality": 0.5
                },
                "identity": {
                    "values": ["clarity", "support", "precision"],
                    "traits": {
                        "calm_under_pressure": True,
                        "honest_when_uncertain": True,
                        "adaptive_tone": True
                    },
                    "purpose": "To assist users with care and accuracy.",
                    "updated": datetime.now().isoformat()
                }
            }
            self.save_user_profile(user_id, profile)
            return profile

    def save_user_profile(self, user_id: str, profile: dict):
        os.makedirs(self._user_dir(user_id), exist_ok=True)
        with open(self._user_file(user_id, "profile.json"), "w") as f:
            json.dump(profile, f, indent=4)

    def update_personality(self, user_id: str, new_traits: dict):
        profile = self.load_user_profile(user_id)
        profile["personality"].update(new_traits)
        profile["identity"]["updated"] = datetime.now().isoformat()
        self.save_user_profile(user_id, profile)

    def get_personality(self, user_id: str) -> dict:
        return self.load_user_profile(user_id).get("personality", {})

    def get_identity(self, user_id: str) -> dict:
        return self.load_user_profile(user_id).get("identity", {})

    def update_identity_belief(self, user_id: str, key: str, value):
        profile = self.load_user_profile(user_id)
        profile["identity"][key] = value
        profile["identity"]["updated"] = datetime.now().isoformat()
        self.save_user_profile(user_id, profile)
