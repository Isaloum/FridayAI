# ======================================
# File: EmpathyInhibitor.py
# Purpose: Blocks emotional overreactions unless context and vector tone demand it.
# ======================================

from typing import Dict
from sentence_transformers import SentenceTransformer, util

class EmpathyInhibitor:
    def __init__(self, threshold: float = 0.7):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.threshold = threshold
        self.false_negative_probes = [
            "not so bad",
            "not too bad",
            "i'm fine",
            "could be worse",
            "i'm okay",
            "not terrible"
        ]
        self.probe_embeddings = self.model.encode(self.false_negative_probes, convert_to_tensor=True)

    def should_block(self, text: str) -> bool:
        input_embedding = self.model.encode(text, convert_to_tensor=True)
        similarity_scores = util.pytorch_cos_sim(input_embedding, self.probe_embeddings)[0]
        max_similarity = similarity_scores.max().item()
        return max_similarity > self.threshold


# === EXAMPLE USAGE ===
if __name__ == "__main__":
    inhibitor = EmpathyInhibitor()
    test_phrases = [
        "not so bad",
        "today was awful",
        "i'm kinda okay",
        "honestly it sucked"
    ]
    for phrase in test_phrases:
        blocked = inhibitor.should_block(phrase)
        print(f"\n'{phrase}' → {'BLOCK empathy' if blocked else 'ALLOW empathy'}")
