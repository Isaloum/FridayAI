# =====================================
# UserPersonaClassifier.py – Adaptive User Type Recognition Engine
# =====================================

from typing import Dict
import re

class UserPersonaClassifier:
    """
    Analyzes user language patterns, expressions, and tone to infer likely persona types.
    Guides Friday’s adaptation in tone, complexity, and empathy format.
    """

    def __init__(self):
        # Basic keywords and expression heuristics for persona inference
        self.persona_rules = [
            {"label": "teenager", "patterns": ["idk", "lol", "bruh", "💀", "deadass", r"\byo\b", "fr"]},
            {"label": "engineer", "patterns": ["debug", "compile", "function", "system", "module", "syntax"]},
            {"label": "spiritual", "patterns": ["divine", "faith", "universe", "higher power", "soul", "chakra"]},
            {"label": "mother", "patterns": ["my son", "my daughter", "school drop off", "diapers", "naptime"]},
            {"label": "elder", "patterns": ["in my day", "back then", "war", "retirement", "my knees"]},
            {"label": "academic", "patterns": ["dissertation", "epistemology", "literature review", "thesis"]},
            {"label": "artist", "patterns": ["palette", "canvas", "brush", "emotion", "abstract"]},
            {"label": "gamer", "patterns": ["gg", "ranked", "grind", "fps", "nerf", "noob"]},
            {"label": "philosopher", "patterns": ["existence", "meaning", "suffering", "nihilism", "stoic"]}
        ]

    def classify(self, user_input: str, tone_vector: Dict = None) -> Dict:
        """
        Determines persona type(s) from input string and emotional tone (optional).
        Returns primary type and tone-adjustment suggestions.
        """
        input_lower = user_input.lower()
        detected = []

        for rule in self.persona_rules:
            for pat in rule["patterns"]:
                if re.search(pat, input_lower):
                    detected.append(rule["label"])
                    break

        primary = detected[0] if detected else "general"

        tone_mod = self._tone_adjustments_for(primary)

        return {
            "persona": primary,
            "matches": detected,
            "tone_profile_modifiers": tone_mod
        }

    def _tone_adjustments_for(self, label: str) -> Dict:
        """
        Suggest tone tweaks based on persona category
        """
        return {
            "teenager": {"formality": 0.2, "humor": 0.9, "warmth": 0.6},
            "engineer": {"formality": 0.7, "precision": 0.9, "warmth": 0.4},
            "spiritual": {"formality": 0.4, "warmth": 0.9, "poetic": 0.8},
            "elder": {"formality": 0.8, "warmth": 0.85, "humor": 0.2},
            "mother": {"warmth": 0.95, "formality": 0.6},
            "academic": {"formality": 0.95, "precision": 0.85, "warmth": 0.5},
            "artist": {"warmth": 0.9, "precision": 0.3, "poetic": 0.95},
            "gamer": {"humor": 0.8, "precision": 0.4, "casual": 0.8},
            "philosopher": {"formality": 0.7, "introspection": 0.95},
            "general": {"warmth": 0.6, "formality": 0.5}
        }.get(label, {"warmth": 0.5, "formality": 0.5})


# =====================
# CLI Test Mode
# =====================
if __name__ == "__main__":
    print("\n🧠 UserPersonaClassifier Test Console")

    classifier = UserPersonaClassifier()

    while True:
        try:
            text = input("You: ").strip()
            if text.lower() in ["exit", "quit"]:
                break

            result = classifier.classify(text)
            print("\n🔍 Persona Inference Result")
            print(f"Primary Persona: {result['persona']}")
            print(f"Matches: {result['matches']}")
            print(f"Tone Modifiers: {result['tone_profile_modifiers']}")
            print("-" * 40)

        except KeyboardInterrupt:
            print("\nSession ended.")
            break
