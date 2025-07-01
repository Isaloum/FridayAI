# =====================================
# NeuralUserPersonaClassifier.py – Embedding-Based Identity Detection
# =====================================

import warnings
import logging

# Suppress noisy NLP library warnings
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from sentence_transformers import SentenceTransformer, util
from typing import Dict, List
import torch
import re

class NeuralUserPersonaClassifier:
    """
    Uses semantic sentence embeddings to classify user identity
    based on tone, phrasing, and contextual behavior.
    """

    def __init__(self):
        # Load semantic similarity model
        self.model = SentenceTransformer("all-mpnet-base-v2")

        # Define example phrases (anchors) for known persona types
        self.anchors = {
            "teenager": [
                "lol idk tbh life’s weird 💀", "bruh I'm not even mad, just vibing", "why school so dumb lmao"
            ],
            "engineer": [
                "I’ve been debugging this module for hours.", "Let's optimize the logic in this control system.", "The architecture needs to scale better."
            ],
            "spiritual": [
                "The universe always brings balance.", "I meditate to align my soul.", "Faith leads me through uncertainty."
            ],
            "elder": [
                "Back in my day, we fixed things ourselves.", "I've seen the world change a lot.", "Life used to be simpler, but harder."
            ],
            "academic": [
                "The epistemological framework lacks internal coherence.", "My thesis explores symbolic meaning in postmodern literature.", "I'm finalizing my dissertation revisions."
            ],
            "artist": [
                "Color expresses what language cannot.", "My brush speaks what I can’t verbalize.", "I painted my grief into abstraction."
            ],
            "gamer": [
                "gg ez noob", "this build is OP in ranked", "nerf this champ now omg"
            ],
            "philosopher": [
                "Suffering gives life its texture.", "Nothing is truly knowable.", "Existence is a recursive hallucination."
            ]
        }

        # Precompute anchor embeddings for efficient comparison
        self.anchor_embeddings = self._build_anchor_embeddings()
        

    def _build_anchor_embeddings(self) -> Dict[str, torch.Tensor]:
        # Precompute sentence embeddings for each anchor group
        return {
            label: self.model.encode(samples, convert_to_tensor=True)
            for label, samples in self.anchors.items()
        }

    def classify(self, user_input: str) -> Dict:
        # Encode user input and compare to each anchor set
        input_embedding = self.model.encode(user_input, convert_to_tensor=True)

        scores = {}
        for label, anchor_vecs in self.anchor_embeddings.items():
            cosine_scores = util.cos_sim(input_embedding, anchor_vecs)
            avg_score = torch.mean(cosine_scores).item()
            scores[label] = avg_score

        # Rank by highest similarity
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_label, best_score = sorted_scores[0]

        return {
            "persona": best_label,
            "confidence": round(best_score, 4),
            "ranking": sorted_scores
        }

    def infer_name(self, text: str) -> str:
        """
        Extracts name from statements like "I'm John", "Call me Sarah"
        """
        match = re.search(r"(i[’']?m|my name is|call me)\s+(\w+)", text.lower())
        if match:
            return match.group(2).capitalize()
        return ""

    def infer_name_confidence(self, text: str) -> (str, float):
        """
        Infer likely user identity or name with fallback confidence.
        """
        name = self.infer_name(text)
        if name:
            return name, 0.95

        classified = self.classify(text)
        label = classified["persona"]
        confidence = classified["confidence"]

        persona_to_name = {
            "engineer": "jake",
            "philosopher": "noah",
            "academic": "sophia",
            "teenager": "zoe",
            "artist": "mira",
            "gamer": "kai",
            "elder": "george",
            "spiritual": "luna"
        }
        guessed_name = persona_to_name.get(label, f"anon_{label}")
        return guessed_name, confidence
           
            
# =====================
# CLI Test Mode
# =====================
if __name__ == "__main__":
    print("\n🧠 NeuralUserPersonaClassifier Test Console")

    clf = NeuralUserPersonaClassifier()

    while True:
        try:
            text = input("You: ").strip()
            if text.lower() in ["exit", "quit"]:
                break

            result = clf.classify(text)
            print("\n🔍 Inferred Persona:")
            print(f"Primary: {result['persona']} ({result['confidence']*100:.1f}%)")
            print("Ranking:")
            for label, score in result["ranking"]:
                print(f" - {label}: {score:.3f}")
            print("-" * 40)

        except KeyboardInterrupt:
            print("\nSession ended.")
            break
