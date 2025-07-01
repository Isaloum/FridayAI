# ==============================================
# File: core/pregnancy/NutritionAgent.py
# Purpose: Suggest trimester-aware, preference-sensitive nutrition tips
# ==============================================

class NutritionAgent:
    def __init__(self, profile):
        self.profile = profile

    def suggest_nutrition(self):
        ctx = self.profile.get_context()
        trimester = ctx.get("trimester")
        preferences = ctx.get("preferences", {})

        tips = []

        # Trimester-based suggestions
        if trimester == "first":
            tips.append("Eat small, frequent meals to combat nausea.")
            tips.append("Focus on folic acid, found in leafy greens and citrus.")
        elif trimester == "second":
            tips.append("Increase calcium and iron intake with dairy and lean meats.")
            tips.append("Add fiber-rich foods to support digestion.")
        elif trimester == "third":
            tips.append("Include healthy fats like avocado for baby brain development.")
            tips.append("Stay hydrated and reduce salty foods to prevent swelling.")

        # Preference-based modifiers
        if preferences.get("nutrition_focus"):
            tips.append("Consider tracking meals in a simple log to spot patterns.")
        if preferences.get("mental_health_focus"):
            tips.append("Avoid caffeine and sugar spikes — they can affect your mood.")

        return tips
