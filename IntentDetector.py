# IntentDetector.py – Semantic Intent Embedding Detector

from sentence_transformers import SentenceTransformer, util

class IntentDetector:
    """
    Uses SentenceTransformer to semantically compare user input against predefined intent descriptions.
    Returns a vector profile with similarity scores per intent.
    """

    def __init__(self):
        self.model = SentenceTransformer("all-mpnet-base-v2")

        # Intent descriptions define conceptual meaning
        self.intent_descriptions = {
            "reflect": "The user wants to reflect on their emotions or behavior.",
            "journal": "The user wants to log a personal moment or share what happened.",
            "task": "The user wants to plan, track, or set reminders for tasks.",
            "learn": "The user is asking to learn or understand something.",
            "ask_ai": "The user wants to know what the AI remembers or has learned.",
            "vent": "The user is emotionally overwhelmed or needs to express frustration.",
            "log_symptom": "The user is describing a physical symptom or health issue.",
            "identity": "The user is asking who Friday is or about her self-awareness.",
            "goal_oriented": "The user is focused on their goals or mission.",
            "progress_reflection": "The user is reflecting on their growth or direction.",
            "identity_check": "The user is questioning their core beliefs or identity.",
            "emotional_dump": "The user is overwhelmed emotionally and venting everything.",
            "observation_log": "The user is passively observing and recording details.",
            "health_monitoring": "The user is tracking or noting physical/mental symptoms.",
            "curiosity": "The user is showing curiosity and wants to explore or learn."
        }

        # Precompute intent embeddings
        self.intent_embeddings = {
            intent: self.model.encode(desc, convert_to_tensor=True)
            for intent, desc in self.intent_descriptions.items()
        }

    def vector_profile(self, user_input: str) -> dict:
        user_embedding = self.model.encode(user_input, convert_to_tensor=True)
        profile = {}

        for intent, emb in self.intent_embeddings.items():
            score = util.pytorch_cos_sim(user_embedding, emb).item()
            if score > 0.1:  # Filter noise
                profile[intent] = round(score, 4)

        return profile
