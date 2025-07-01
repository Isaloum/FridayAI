# ================================================
# File: NeuralSchedulerCore.py
# Purpose: Autonomous background thinking system
# ================================================

import random
from datetime import datetime

class NeuralSchedulerCore:
    """
    Background autonomous process that triggers internal events: dreaming, planning, reflection.
    """

    def __init__(self, identity_core, emotion_core, planner_core, narrative_fusion):
        self.identity = identity_core
        self.emotion = emotion_core
        self.planner = planner_core
        self.narrative = narrative_fusion

    def tick(self):
        print("[DEBUG] TICK method is ACTIVE from:", __file__)
        """
        Called periodically (e.g. every minute/hour) to simulate internal activity.
        """
        event_type = random.choice(["dream", "reflect", "plan"])
        now = datetime.now().isoformat()

        if event_type == "dream":
            thought = self.narrative.simulate_internal_event("idle_dream")
            mood = self.emotion.analyze_emotion(thought)
            self.identity.log_event(thought, kind="dream")
            
            if isinstance(mood, dict):
                dominant = mood.get("top_emotion", "neutral")
            else:
                dominant = "neutral"
            self.identity.update_mood(dominant)


            return {
                "event": "dream",
                "content": thought,
                "mood": mood,
                "time": now
            }


        elif event_type == "reflect":
            reflection = f"Reflecting on past emotional drift at {now}."
            self.identity.log_event(reflection, kind="self-reflect")
            return {
                "event": "reflect",
                "content": reflection,
                "time": now
            }

        elif event_type == "plan":
            goal = f"Reassess internal goal priorities."
            self.planner.create_structured_plan("self_restructure", steps=[
                {"type": "text", "note": "Recalibrate beliefs and goals"}
            ])
            return {
                "event": "plan",
                "content": goal,
                "time": now
            }

        return {"event": "none", "time": now}
