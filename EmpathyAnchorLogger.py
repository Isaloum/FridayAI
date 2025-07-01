# =============================================
# File: EmpathyAnchorLogger.py
# Purpose: Create emotional anchors when user shows high emotional depth
# Dependencies: EmotionalAnchorCore
# =============================================

from datetime import datetime
from core.EmotionalAnchorCore import EmotionalAnchorCore

class EmpathyAnchorLogger:
    def __init__(self, anchor_core: EmotionalAnchorCore):
        self.anchor_core = anchor_core

    def log_anchor_if_deep(self, user_id: str, user_input: str, inferred: dict, strategy: dict):
        """
        Automatically logs an emotional anchor if risk_level is medium or high.
        This affects Friday's future tone via EmotionalAnchorCore.
        """
        if not inferred or inferred.get("risk_level") not in ["medium", "high"]:
            return  # Nothing to log

        anchor_id = f"{user_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        emotion = inferred.get("primary_emotion", "unknown")
        style = strategy.get("strategy_tag", "empathetic")
        risk = inferred["risk_level"]

        self.anchor_core.add_anchor(
            anchor_id=anchor_id,
            description=f"Detected {emotion} with {risk} risk",
            emotion_type=emotion,
            memory_id=None,
            impact_weight=0.8 if risk == "medium" else 1.0,
            tone_shift_map={
                style: 0.5,
                "precise": -0.2 if risk == "high" else 0.0
            }
        )
        print(f"📌 [AnchorLogger] Emotion anchor saved: {anchor_id}")
