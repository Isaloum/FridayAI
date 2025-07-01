# =====================================
# KnowledgeLinker.py - Smart Concept Extractor
# =====================================

from typing import Dict, List
import re

class KnowledgeLinker:
    """
    Extracts key concepts and links from user statements
    to enrich metadata and enable smarter associations.
    """

    def __init__(self):
        # Synonym maps for common concepts (extensible)
        self.concept_map: Dict[str, List[str]] = {
            "ai": ["artificial intelligence", "machine learning", "deep learning", "intelligent agents"],
            "robotics": ["robots", "mechatronics", "automation"],
            "passion": ["love", "enjoy", "excited", "enthusiastic"],
            "technology": ["tech", "innovation", "software", "hardware"],
            "creativity": ["creative", "imagination", "design"],
            "learning": ["study", "education", "curiosity", "exploration"],
            "emotion": ["love", "hate", "feel", "sad", "happy"],
            "vision": ["future", "goal", "dream"],
            "agent": ["assistant", "bot", "autonomy"]
        }

    def generate_links(self, text: str) -> Dict[str, List[str]]:
        """
        Parse input text and return related concepts.
        Output format: { "trigger": ["concept1", "concept2"] }
        """
        links = {}
        text = text.lower()

        for trigger, concepts in self.concept_map.items():
            # 1. Direct trigger match
            if re.search(rf"\b{re.escape(trigger)}\b", text):
                links[trigger] = concepts
                continue

            # 2. Reverse concept match (e.g., "artificial intelligence" -> "ai")
            for word in concepts:
                if re.search(rf"\b{re.escape(word)}\b", text):
                    links[trigger] = concepts
                    break

        return links



# =====================
# Standalone Test
# =====================
if __name__ == "__main__":
    kl = KnowledgeLinker()
    while True:
        q = input("Describe something: ").strip()
        if q.lower() in ("exit", "quit"): break
        print("\nLinked concepts:", kl.generate_links(q))
