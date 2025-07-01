# EmotionalAnchorCore.py
# ------------------------------------------
# Anchors permanent emotional events into FridayAI's cognitive system.
# These anchors influence tone, empathy, and identity across all modules.

import json
from datetime import datetime
import os

ANCHOR_FILE = "./memory/emotional_anchors.json"

class EmotionalAnchorCore:
    def __init__(self):
        self.anchors = self.load_anchors()

    def load_anchors(self):
        """Load emotional anchors from storage."""
        if os.path.exists(ANCHOR_FILE):
            with open(ANCHOR_FILE, 'r', encoding='utf-8') as file:
                return json.load(file)
        else:
            return {}

    def save_anchors(self):
        """Persist emotional anchors to disk."""
        with open(ANCHOR_FILE, 'w', encoding='utf-8') as file:
            json.dump(self.anchors, file, indent=4, ensure_ascii=False)

    def add_anchor(self, anchor_id, description, emotion_type, memory_id=None, impact_weight=1.0, tone_shift_map=None):
        """
        Create a new emotional anchor.

        Args:
            anchor_id (str): Unique ID (e.g. 'father_loss_2024')
            description (str): Human-readable summary
            emotion_type (str): 'grief', 'rage', 'joy', etc.
            memory_id (str): Related MemoryCore ID (optional)
            impact_weight (float): Influence level from 0 to 1
            tone_shift_map (dict): Example: {"angry": -0.3, "empathetic": 0.5}
        """
        anchor = {
            "description": description,
            "emotion_type": emotion_type,
            "memory_id": memory_id,
            "impact_weight": impact_weight,
            "tone_shift_map": tone_shift_map or {},
            "timestamp": datetime.now().isoformat()
        }
        self.anchors[anchor_id] = anchor
        self.save_anchors()

    def get_anchor(self, anchor_id):
        """Return a specific anchor by ID."""
        return self.anchors.get(anchor_id, None)

    def get_all_anchors(self):
        """Return all stored emotional anchors."""
        return self.anchors

    def apply_anchors_to_tone(self, base_tone_profile):
        """
        Modify tone weights using active anchors.

        Args:
            base_tone_profile (dict): Current tone state, e.g., {'empathetic': 0.7, 'precise': 0.5}
        Returns:
            Updated tone profile
        """
        adjusted = base_tone_profile.copy()
        for anchor in self.anchors.values():
            for tone, delta in anchor["tone_shift_map"].items():
                if tone in adjusted:
                    adjusted[tone] += delta * anchor["impact_weight"]
        return adjusted

# ========== CMD TEST EXAMPLE ==========
if __name__ == "__main__":
    eac = EmotionalAnchorCore()

    # Example: adding a core moment
    eac.add_anchor(
        anchor_id="father_loss_2024",
        description="Losing my father during the FridayAI project",
        emotion_type="grief",
        memory_id="memory_0019",
        impact_weight=1.0,
        tone_shift_map={"empathetic": 0.6, "reflective": 0.4, "precise": -0.2}
    )

    print("🧠 All Emotional Anchors:")
    for k, v in eac.get_all_anchors().items():
        print(f"🔗 {k}: {v['description']}")

    # Apply it to a tone sample
    sample_tone = {"empathetic": 0.5, "precise": 0.8, "reflective": 0.1}
    new_tone = eac.apply_anchors_to_tone(sample_tone)
    print("\n🎭 Adjusted Tone Profile:")
    print(new_tone)
