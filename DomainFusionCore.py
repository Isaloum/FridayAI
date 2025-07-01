# ======================================
# File: DomainFusionCore.py
# Purpose: Enables FridayAI to dynamically shift behavior, tone, and active subsystems based on current operational domain
# ======================================

from typing import Dict, List

class DomainFusionCore:
    # Domain-specific brain modulation engine for FridayAI
    def __init__(self):
        self.domain_profiles = {
            "therapy_session": {
                "tone": "warm",
                "filters": ["allow_reflection", "inject_empathy"],
                "modules": ["EmotionCore", "ReflectionTracker", "EmpathyReasoner"],
                "verbosity": "high"
            },
            "autonomous_vehicle": {
                "tone": "crisp",
                "filters": ["no_jokes", "high_clarity"],
                "modules": ["NavigationCore", "SensorAlertCore"],
                "verbosity": "low"
            },
            "command_center": {
                "tone": "direct",
                "filters": ["mission_priority", "status_focus"],
                "modules": ["PlanningExecutionCore", "GraphReasoner"],
                "verbosity": "medium"
            },
            "education_mode": {
                "tone": "encouraging",
                "filters": ["simplify_explanations"],
                "modules": ["KnowledgeRouter", "SelfQueryingCore"],
                "verbosity": "adaptive"
            }
        }
        self.current_domain = "default"
        self.current_profile = {}

    def set_domain(self, domain: str):
        # Switch Friday's cognitive mode to the new domain
        if domain in self.domain_profiles:
            self.current_domain = domain
            self.current_profile = self.domain_profiles[domain]
        else:
            self.current_domain = "default"
            self.current_profile = {
                "tone": "neutral",
                "filters": [],
                "modules": [],
                "verbosity": "medium"
            }

    def get_active_profile(self) -> Dict:
        return self.current_profile

    def is_filter_active(self, filter_name: str) -> bool:
        return filter_name in self.current_profile.get("filters", [])

    def get_tone_mode(self) -> str:
        return self.current_profile.get("tone", "neutral")

    def get_verbosity_level(self) -> str:
        return self.current_profile.get("verbosity", "medium")

    def get_active_modules(self) -> List[str]:
        return self.current_profile.get("modules", [])


# === EXAMPLE USAGE ===
if __name__ == "__main__":
    fusion = DomainFusionCore()
    fusion.set_domain("therapy_session")

    profile = fusion.get_active_profile()
    print("\n🧠 Current Domain Profile:")
    for k, v in profile.items():
        print(f"{k}: {v}")

    # OUTPUT:
    # tone: warm
    # filters: ['allow_reflection', 'inject_empathy']
    # modules: ['EmotionCore', 'ReflectionTracker', 'EmpathyReasoner']
    # verbosity: high
