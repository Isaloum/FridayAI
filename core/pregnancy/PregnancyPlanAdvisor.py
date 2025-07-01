# core/pregnancy/PregnancyPlanAdvisor.py

from datetime import datetime

class PregnancyPlanAdvisor:
    def __init__(self, memory, trimester=None):
        self.memory = memory
        self.trimester = trimester

    def suggest_goals(self):
        if not self.trimester:
            return ["Track how many weeks pregnant you are."]

        plans = {
            "first": [
                "Schedule your first prenatal checkup.",
                "Start a gentle walking routine.",
                "Begin a food/mood diary."
            ],
            "second": [
                "Start gentle stretching or yoga.",
                "Document positive body changes.",
                "Discuss support systems with partner."
            ],
            "third": [
                "Pack your hospital bag.",
                "Practice breathing techniques.",
                "Log things that bring you comfort."
            ]
        }

        return plans.get(self.trimester, ["Reflect on your well-being today."])

    def save_goal_suggestions(self, goals: list):
        timestamp = datetime.now().isoformat()
        self.memory.save_memory({
            "type": "pregnancy_goals",
            "trimester": self.trimester,
            "goals": goals,
            "timestamp": timestamp
        })
