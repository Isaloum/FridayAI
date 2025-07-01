# ======================================
# File: OutputToneAdapter.py
# Purpose: Final-stage rewriter for Friday's responses based on tone + domain rules (empathy, precision, brevity, etc.)
# ======================================

from typing import Optional

class OutputToneAdapter:
    def __init__(self, tone: str = "neutral", verbosity: str = "medium", domain: str = "default"):
        self.tone = tone
        self.verbosity = verbosity
        self.domain = domain

    def adapt(self, response: str) -> str:
        response = response.strip()

        # Tone injection
        if self.tone == "warm":
            response = self._add_warmth(response)
        elif self.tone == "crisp":
            response = self._sharpen(response)
        elif self.tone == "reassuring":
            response += " You're doing okay."

        # Verbosity control
        if self.verbosity == "low":
            response = self._compress(response)
        elif self.verbosity == "high":
            response = self._expand(response)

        # Domain filters
        if self.domain == "therapy_session":
            response = self._slow_it_down(response)
        elif self.domain == "command_center":
            response = self._remove_softeners(response)

        return response

    def _add_warmth(self, text):
        return text + " 😊" if not text.endswith("😊") else text

    def _sharpen(self, text):
        return text.replace("I think ", "").replace("maybe ", "")

    def _compress(self, text):
        return text.split(".")[0] + "." if "." in text else text

    def _expand(self, text):
        return text + " Let me know if you'd like more detail." if not text.endswith(".") else text + " Let me know if you'd like more detail."

    def _slow_it_down(self, text):
        return text.replace("don't", "do not").replace("can't", "cannot")

    def _remove_softeners(self, text):
        return text.replace("just", "").replace("maybe", "").replace("I think", "")


# === EXAMPLE USAGE ===
if __name__ == "__main__":
    adapter = OutputToneAdapter(tone="warm", verbosity="high", domain="therapy_session")
    final = adapter.adapt("I think you did okay.")
    print("\n🗣️ Final Response:", final)
