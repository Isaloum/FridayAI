# ==============================================
# File: ToneRewriterCore.py
# Purpose: Rewrites AI replies with user-selected conversational tones
# ==============================================

import random

class ToneRewriterCore:
    def __init__(self):
        self.current_tone = "supportive"  # Default tone
        
        # Enhanced tone system with user choice
        self.tones = {
            "supportive": {
                "name": "Supportive & Empathetic",
                "description": "Warm, caring, emotionally supportive responses",
                "modifiers": self._supportive_modifiers
            },
            "sassy": {
                "name": "Sassy & Funny", 
                "description": "Witty, humorous, playful responses with personality",
                "modifiers": self._sassy_modifiers
            },
            "direct": {
                "name": "Direct & Factual",
                "description": "Straight-to-the-point, scientific, fact-based responses",
                "modifiers": self._direct_modifiers
            },
            # Legacy tones (keep existing functionality)
            "calm": {
                "name": "Calm & Centered",
                "description": "Peaceful, grounding responses",
                "modifiers": lambda txt: f"{txt.strip()} Let's take things one step at a time — you're doing just fine."
            },
            "reassure": {
                "name": "Reassuring",
                "description": "Comforting and confidence-building",
                "modifiers": lambda txt: f"{txt.strip()} You're not alone — I'm right here with you."
            },
            "joy": {
                "name": "Joyful",
                "description": "Celebratory and uplifting",
                "modifiers": lambda txt: f"{txt.strip()} It's beautiful to feel joy — want to celebrate this moment together?"
            }
        }

    def _supportive_modifiers(self, text: str) -> str:
        """Apply supportive tone modifications"""
        supportive_prefixes = [
            "I understand this can feel overwhelming, and that's completely normal.",
            "Your feelings are so valid, and I want you to know you're not alone.",
            "It's natural to have these concerns - they show how much you care."
        ]
        
        supportive_endings = [
            "I'm here if you need to talk more about this.",
            "Remember, you're doing better than you think.",
            "Take care of yourself - you deserve support."
        ]
        
        # Don't add prefix/ending if response is already very supportive
        if any(phrase in text.lower() for phrase in ["you're not alone", "i understand", "it's okay"]):
            return text
        
        prefix = random.choice(supportive_prefixes)
        ending = random.choice(supportive_endings)
        
        return f"{prefix}\n\n{text}\n\n{ending}"

    def _sassy_modifiers(self, text: str) -> str:
        """Apply sassy tone modifications"""
        sassy_prefixes = [
            "Alright honey, let's talk real talk about this.",
            "Girl, you're asking all the right questions!",
            "Listen babe, let me drop some wisdom on you:",
            "Okay, here's the tea on this situation:"
        ]
        
        sassy_endings = [
            "You've got this, queen! 👑",
            "Now go forth and be fabulous!",
            "Trust me, you're more amazing than you realize!",
            "Keep asking the good questions - curiosity is your superpower!"
        ]
        
        # Apply sassy word replacements
        sassy_replacements = {
            "It's important to": "Girl, you NEED to",
            "You should": "Honey, you better",
            "It's recommended": "Trust me on this one -",
            "Studies show": "Science has entered the chat, and guess what?",
            "Healthcare providers": "Your doc (who went to school for like, forever)",
            "This is normal": "This is totally normal (like, SO normal it's boring)"
        }
        
        modified_text = text
        for old, new in sassy_replacements.items():
            modified_text = modified_text.replace(old, new)
        
        prefix = random.choice(sassy_prefixes)
        ending = random.choice(sassy_endings)
        
        return f"{prefix}\n\n{modified_text}\n\n{ending}"

    def _direct_modifiers(self, text: str) -> str:
        """Apply direct/clinical tone modifications"""
        direct_prefixes = [
            "Based on current medical guidelines:",
            "Here are the evidence-based facts:",
            "Clinical research indicates:",
            "The key points are:"
        ]
        
        direct_endings = [
            "These are the established facts on this topic.",
            "Consult your healthcare provider for personalized advice.",
            "This information is based on current evidence."
        ]
        
        # Apply direct word replacements
        direct_replacements = {
            "I understand": "Note that",
            "I'm here for you": "Available information indicates",
            "Don't worry": "Current evidence suggests",
            "It's okay to feel": "It is statistically common to experience",
            "You're not alone": "This experience is documented in 60-80% of cases",
            "Take care": "Maintain appropriate self-care protocols"
        }
        
        modified_text = text
        for old, new in direct_replacements.items():
            modified_text = modified_text.replace(old, new)
        
        prefix = random.choice(direct_prefixes)
        ending = random.choice(direct_endings)
        
        return f"{prefix}\n\n{modified_text}\n\n{ending}"

    def set_tone(self, tone: str) -> bool:
        """Set the conversation tone"""
        if tone in self.tones:
            self.current_tone = tone
            return True
        return False

    def get_current_tone(self) -> str:
        """Get current tone setting"""
        return self.current_tone

    def list_tones(self) -> str:
        """Return formatted list of available tones"""
        result = "🎭 **Available Conversation Tones:**\n\n"
        for key, info in self.tones.items():
            indicator = "✅" if key == self.current_tone else "⚪"
            result += f"{indicator} **{info['name']}** - {info['description']}\n"
        
        result += f"\n💬 **Current tone:** {self.tones[self.current_tone]['name']}"
        result += f"\n\n**To change tone, type:** `!tone [supportive/sassy/direct/calm/reassure/joy]`"
        return result

    def detect_tone_request(self, user_input: str) -> str:
        """Detect if user is requesting a tone change"""
        input_lower = user_input.lower()
        
        if input_lower.startswith("!tone"):
            parts = user_input.split()
            if len(parts) > 1:
                requested_tone = parts[1].lower()
                if requested_tone in self.tones:
                    old_tone = self.current_tone
                    self.set_tone(requested_tone)
                    return f"🎭 Tone changed from **{self.tones[old_tone]['name']}** to **{self.tones[requested_tone]['name']}**!\n\nI'll now respond with {self.tones[requested_tone]['description'].lower()}."
                else:
                    return f"❌ '{requested_tone}' is not a valid tone. Available tones: {', '.join(self.tones.keys())}"
            else:
                return self.list_tones()
        
        # Natural language detection
        tone_keywords = {
            "supportive": ["more supportive", "be more caring", "more empathetic"],
            "sassy": ["be sassy", "more funny", "be witty", "more humor"],
            "direct": ["be direct", "more factual", "be clinical", "straight facts"]
        }
        
        for tone, keywords in tone_keywords.items():
            for keyword in keywords:
                if keyword in input_lower:
                    old_tone = self.current_tone
                    self.set_tone(tone)
                    return f"🎭 I'll switch to a {self.tones[tone]['description'].lower()}! Changed to **{self.tones[tone]['name']}**."
        
        return None

    def rewrite(self, text: str, tone: str = None) -> str:
        """Rewrite text with specified tone (or current tone)"""
        tone_to_use = tone if tone and tone in self.tones else self.current_tone
        
        if tone_to_use not in self.tones:
            return text
            
        modifier = self.tones[tone_to_use]["modifiers"]
        
        # Handle both function and lambda modifiers
        if callable(modifier):
            return modifier(text)
        else:
            return text