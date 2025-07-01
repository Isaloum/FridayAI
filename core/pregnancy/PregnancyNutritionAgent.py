# ==============================================
# File: core/pregnancy/PregnancyNutritionAgent.py
# Purpose: Suggest trimester-specific diet tips personalized to user profile
# ==============================================

class PregnancyNutritionAgent:
    def __init__(self, profile):
        self.profile = profile

    def get_diet_tips(self):
        ctx = self.profile.get_context()
        trimester = ctx.get("trimester", "unknown")
        preferences = ctx.get("preferences", {})
        health_flags = ctx.get("health_flags", [])

        tips = {
            "first": [
                "Eat small, frequent meals to combat nausea.",
                "Include folate-rich foods like leafy greens and citrus."
            ],
            "second": [
                "Boost calcium and vitamin D for bone development.",
                "Eat fiber-rich foods to support digestion."
            ],
            "third": [
                "Stay hydrated and manage heartburn with gentle meals.",
                "Get enough iron through lean meats and legumes."
            ]
        }

        special_notes = []

        if "gestational_diabetes" in health_flags:
            special_notes.append("Monitor sugar intake — choose whole grains over refined carbs.")

        if preferences.get("nutrition_focus"):
            special_notes.append("You're doing great prioritizing nutrition! Keep a meal journal if possible.")

        base = tips.get(trimester, ["Eat a balanced, colorful diet with plenty of hydration."])
        return base + special_notes
