# ======================================
# File: DomainAwareResponseAdapter.py
# Purpose: Shapes FridayAI's final responses by fusing tone, verbosity, and domain-specific filters from DomainFusionCore
# ======================================

from typing import Dict

class DomainAwareResponseAdapter:
    # Final-stage adapter to modulate responses based on active domain profile
    def __init__(self, domain_core):
        self.domain_core = domain_core

    def adapt_response(self, raw_text: str) -> str:
        tone = self.domain_core.get_tone_mode()
        verbosity = self.domain_core.get_verbosity_level()
        filters = self.domain_core.get_active_profile().get("filters", [])

        response = raw_text.strip()

        # === Tone shaping ===
        if tone == "warm":
            response = self._apply_warmth(response)
        elif tone == "crisp":
            response = self._tighten(response)
        elif tone == "encouraging":
            response += " You're doing great."

        # === Verbosity shaping ===
        if verbosity == "low":
            response = self._compress(response)
        elif verbosity == "high":
            response = self._expand(response)

        # === Filter logic ===
        if "no_jokes" in filters:
            response = self._strip_humor(response)
        if "simplify_explanations" in filters:
            response = self._simplify(response)

        return response

    def _apply_warmth(self, text: str) -> str:
        return text.replace(".", ". 😊") if "." in text else text + " 😊"

    def _tighten(self, text: str) -> str:
        return text.replace("I think ", "").replace("maybe ", "")

    def _compress(self, text: str) -> str:
        return text.split(".")[0] + "." if "." in text else text

    def _expand(self, text: str) -> str:
        return text + " If you’d like, I can explain more." if not text.endswith(".") else text + " If you’d like, I can explain more."

    def _strip_humor(self, text: str) -> str:
        return text.replace("😂", "").replace("lol", "")

    def _simplify(self, text: str) -> str:
        return text.replace("contemplate", "think").replace("analyze", "look at")


# === EXAMPLE USAGE ===
if __name__ == "__main__":
    from DomainFusionCore import DomainFusionCore

    fusion = DomainFusionCore()
    fusion.set_domain("therapy_session")
    adapter = DomainAwareResponseAdapter(fusion)

    raw = "You should consider reflecting on what you truly need."
    shaped = adapter.adapt_response(raw)

    print("\n🗣️ Domain-Aware Response:", shaped)

    # OUTPUT: You should consider reflecting on what you truly need. 😊 If you’d like, I can explain more.
