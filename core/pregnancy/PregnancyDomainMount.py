# ===============================================
# File: core/pregnancy/PregnancyDomainMount.py
# Purpose: Domain-level composition for pregnancy features
# ===============================================

from core.pregnancy.PregnancySupportCore import PregnancySupportCore
from core.pregnancy.PregnancyPlanAdvisor import PregnancyPlanAdvisor
from core.pregnancy.PregnancyJournalInterface import PregnancyJournalInterface
from core.pregnancy.PregnancyReflectionEngine import PregnancyReflectionEngine
from core.pregnancy.PregnancyEmotionDriftAnalyzer import PregnancyEmotionDriftAnalyzer
from core.pregnancy.PregnancyUserProfile import PregnancyUserProfile  # NEW
from core.pregnancy.PregnancyJourneyAgent import PregnancyJourneyAgent
from core.pregnancy.PregnancyWeeklyDevelopmentAgent import PregnancyWeeklyDevelopmentAgent


class PregnancyDomainMount:
    def __init__(self, memory, emotion_core, identity):
        self.profile = PregnancyUserProfile()  # NEW
        self.support = PregnancySupportCore(memory, emotion_core, identity, self.profile)
        self.journal = PregnancyJournalInterface(memory, emotion_core, identity)
        self.advisor = PregnancyPlanAdvisor(memory, self.profile)
        self.reflection = PregnancyReflectionEngine(memory)
        self.emotion_drift = PregnancyEmotionDriftAnalyzer(memory)
        self.journey = PregnancyJourneyAgent(self.profile, memory)
        self.weekly_dev = PregnancyWeeklyDevelopmentAgent()
        
    def get_abilities(self):
        """
        Return list of abilities this domain provides.
        """
        return [
            "pregnancy_support",
            "emotional_guidance", 
            "health_tracking",
            "milestone_tracking"
        ]
