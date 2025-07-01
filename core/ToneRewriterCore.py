# ==============================================
# File: core/ToneRewriterCore.py
# Purpose: Rewrites AI replies to sound more calm, kind, or joyful
# ==============================================

class ToneRewriterCore:
    def __init__(self):
        # Predefined tone prompts — can be enhanced later
        self.tones = {
            "calm": lambda txt: f"{txt.strip()} Let's take things one step at a time — you're doing just fine.",
            "reassure": lambda txt: f"{txt.strip()} You're not alone — I'm right here with you.",
            "joy": lambda txt: f"{txt.strip()} It's beautiful to feel joy — want to celebrate this moment together?"
        }

    def rewrite(self, text: str, tone: str = "reassure") -> str:
        if tone not in self.tones:
            return text
        return self.tones[tone](text)
