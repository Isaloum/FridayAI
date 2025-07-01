# IntentionReflectionCore.py

class IntentionReflectionCore:
    def __init__(self, memory_core, emotion_core, goal_log):
        self.memory_core = memory_core
        self.emotion_core = emotion_core
        self.goal_log = goal_log

    def cross_reference_goals_with_emotions(self):
        """Find goals where emotional state suggests disengagement or tension"""
        mismatched = []
        for goal in self.goal_log:
            emotion = self.emotion_core.get_emotional_state_for(goal)
            if emotion in ('disengaged', 'anxious', 'flat'):
                mismatched.append(goal)
        return mismatched

    def scan_for_detachment(self):
        """Find goals with no recent memory activity"""
        detached = []
        for goal in self.goal_log:
            if not self.memory_core.has_recent_reference(goal):
                detached.append(goal)
        return detached

    def prompt_realignment(self):
        """Return questions to ask the user for self-check"""
        mismatches = self.cross_reference_goals_with_emotions()
        detaches = self.scan_for_detachment()
        prompts = []
        for goal in mismatches:
            prompts.append(f"You seem emotionally distant from your goal: '{goal}'. Still meaningful?")
        for goal in detaches:
            prompts.append(f"Haven't seen much about '{goal}' lately. Still part of the plan?")
        return prompts

    def identity_cohesion_score(self):
        """Rate how well your current goals align with your emotional + memory state"""
        total = len(self.goal_log)
        if total == 0:
            return 1.0
        misaligned = len(self.cross_reference_goals_with_emotions())
        detached = len(self.scan_for_detachment())
        penalty = misaligned + detached
        score = max(0, 1 - (penalty / total))
        return round(score, 2)
