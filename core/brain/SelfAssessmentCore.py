# ==============================================
# File: core/brain/SelfAssessmentCore.py
# Purpose: Reflect on recent thought pulses
# ==============================================

class SelfAssessmentCore:
    def __init__(self, cognitive_core):
        self.cognitive_core = cognitive_core
        
    def detect_emotion_drift(self, count=3):
        import numpy as np
        pulses = self.cognitive_core.thought_log[-count:]
        if len(pulses) < 2:
            return 0.0
        diffs = []
        for i in range(1, len(pulses)):
            a = np.array(pulses[i-1]["emotion"])
            b = np.array(pulses[i]["emotion"])
            diff = np.linalg.norm(a - b)
            diffs.append(diff)
        return float(np.mean(diffs))

    def summarize_recent_pulses(self, count=3):
        pulses = self.cognitive_core.thought_log[-count:]
        summaries = []
        for p in pulses:
            summaries.append({
                "emotion": p["emotion"],
                "top_memory_summary": p["top_memory"].get("summary", "None") if p["top_memory"] else "None",
                "conflicts": p["conflict_tags"]
            })
        return summaries
        
    def reflection_score(self, count=3):
        pulses = self.cognitive_core.thought_log[-count:]
        score = 0
        for p in pulses:
            if p["conflict_tags"]:
                score -= 1
            else:
                score += 1
        return score / max(len(pulses), 1)
        
    def reflect(self):
        return {
            "drift_score": self.detect_emotion_drift(),
            "reflection_score": self.reflection_score(),
            "recent_pulses": self.summarize_recent_pulses()
        }


if __name__ == "__main__":
    from CognitivePrioritizationCore import CognitivePrioritizationCore

    class MockMemoryCore:
        def get_recent_entries(self, limit=10):
            return [
                {"emotion_vector": [0.1, 0.9], "summary": "Worried about baby", "tags": ["anxiety"]},
                {"emotion_vector": [0.9, 0.1], "summary": "Had a calm day", "tags": ["joy"]}
            ]

    class MockEmotionCore:
        def get_current_emotion(self):
            return [0.1, 0.9]

    cog = CognitivePrioritizationCore(MockMemoryCore(), MockEmotionCore())
    e, m = cog.evaluate_current_state()
    r = cog.rank_memory_relevance(e)
    cog.generate_thought_pulse(r, e)
    cog.generate_thought_pulse(r, e)  # Twice to test history
    assess = SelfAssessmentCore(cog)
    for summary in assess.summarize_recent_pulses():
        print(summary)

    print("--- Self Reflection ---")
    report = assess.reflect()
    for k, v in report.items():
        print(k + ":", v)


