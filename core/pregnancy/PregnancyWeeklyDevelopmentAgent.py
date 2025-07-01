# ==============================================
# File: core/pregnancy/PregnancyWeeklyDevelopmentAgent.py
# Purpose: Generate baby development summary per week
# ==============================================

class PregnancyWeeklyDevelopmentAgent:
    def __init__(self):
        self.data = {
            1: "Fertilization and implantation begin.",
            5: "Heartbeat may be detected via ultrasound.",
            10: "Vital organs are formed.",
            14: "Baby starts moving (you may not feel it yet).",
            20: "Halfway there! Anatomy scan recommended.",
            30: "Baby gains weight rapidly and practices breathing.",
            40: "Full term! Time to prepare for birth."
        }

    def get_update(self, week: int) -> str:
        milestones = [w for w in sorted(self.data) if w <= week]
        if not milestones:
            return "No data available yet for this stage."
        latest = milestones[-1]
        return f"Week {week}: {self.data[latest]}"
