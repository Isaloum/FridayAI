# =====================================
# NLUProcessor.py
# Natural Language Understanding Processor for Friday
# =====================================

import re
from difflib import get_close_matches

class NLUProcessor:
    """Processes natural input and manages memory updates."""

    def __init__(self, memory_core):
        self.memory = memory_core

    def parse(self, user_input):
        """Main entry to understand user input."""
        lower = user_input.lower()

        # Explicit memory commands
        if lower.startswith("remember "):
            return self._handle_remember(lower)
        if lower.startswith("update "):
            return self._handle_update(lower)
        if lower.startswith("delete "):
            return self._handle_delete(lower)
        if lower.startswith("list facts"):
            return self.memory.list_facts()
        if lower.startswith("what is my"):
            return self._handle_query(lower)

        # If casual conversation — sniff for facts
        sniffed_fact = self._sniff_fact(user_input)
        if sniffed_fact:
            return sniffed_fact

        return None  # fallback to GPT if nothing recognized

    def _handle_remember(self, text):
        try:
            key_value = text[9:].split(" is ")
            key, value = key_value[0].strip(), key_value[1].strip()
            return self.memory.save_fact(key, value)
        except:
            return "⚠️ Use format: remember [thing] is [value]."

    def _handle_update(self, text):
        try:
            key_value = text[7:].split(" to ")
            key, value = key_value[0].strip(), key_value[1].strip()
            return self.memory.update_fact(key, value)
        except:
            return "⚠️ Use format: update [thing] to [new value]."

    def _handle_delete(self, text):
        fact = text[7:].strip()
        return self.memory.delete_fact(fact)

    def _handle_query(self, text):
        fact_key = text.split("what is my")[1].strip().replace("?", "")
        fact = self.memory.get_fact(fact_key)
        if not fact:
            closest = self.memory.find_closest_fact(fact_key)
            if closest:
                fact = self.memory.get_fact(closest)
                return f"Your {closest} is {fact}."
            else:
                return f"⚠️ I don't know your {fact_key} yet."
        return f"Your {fact_key} is {fact}."

    def _sniff_fact(self, text):
        """Auto extract facts from casual sentences."""
        patterns = [
            (r"my favorite (.+?) is (.+)", "favorite {}"),
            (r"i live in (.+)", "location"),
            (r"my hobby is (.+)", "hobby"),
            (r"i work as (.+)", "occupation")
        ]

        for pattern, template in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if '{}' in template:
                    field = template.format(match.group(1).strip())
                    value = match.group(2).strip()
                else:
                    field = template
                    value = match.group(1).strip()
                return self.memory.save_fact(field, value)

        return None
