# ==============================================
# File: core/brain/CognitivePrioritizationCore.py
# Purpose: Prioritize thoughts/emotions to drive behavior
# ==============================================

class CognitivePrioritizationCore:
    def __init__(self, memory_core, emotion_core):
        self.memory_core = memory_core
        self.emotion_core = emotion_core
        self.thought_log = []

    def evaluate_current_state(self):
        emotion = self.emotion_core.get_current_emotion()
        memories = self.memory_core.get_recent_entries(limit=10)
        return emotion, memories


    def rank_memory_relevance(self, emotion_vector):
        ranked = []
        for mem in self.memory_core.get_recent_entries(limit=10):
            score = self._cosine_similarity(mem["emotion_vector"], emotion_vector)
            ranked.append((score, mem))
        ranked.sort(reverse=True, key=lambda x: x[0])
        return [mem for score, mem in ranked]

    def _cosine_similarity(self, vec1, vec2):
        import numpy as np
        a = np.array(vec1)
        b = np.array(vec2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


    def detect_conflict(self, priorities):
        tags = {}
        for mem in priorities:
            for tag in mem.get("tags", []):
                tags[tag] = tags.get(tag, 0) + 1

        conflicts = [tag for tag, count in tags.items() if count > 1]
        return conflicts


    def generate_thought_pulse(self, ranked_memories, current_emotion):
        pulse = {
            "emotion": current_emotion,
            "top_memory": ranked_memories[0] if ranked_memories else None,
            "conflict_tags": self.detect_conflict(ranked_memories),
        }
        self.thought_log.append(pulse)
        print("[Thought Pulse]")
        print("Emotion:", current_emotion)
        print("Top Memory:", pulse["top_memory"].get("summary", "None"))
        print("Conflicts:", pulse["conflict_tags"])
        return pulse


    def route_behavior(self):
        if not self.thought_log:
            return "idle"
        pulse = self.thought_log[-1]
        tags = pulse["top_memory"].get("tags", []) if pulse["top_memory"] else []
        if "anxiety" in tags:
            return "mental_health"
        if "joy" in tags:
            return "journal"
        return "default"



if __name__ == "__main__":
    class MockMemoryCore:
        def get_recent_entries(self, limit=10):
            return [
                {"emotion_vector": [0.1, 0.9], "summary": "Worried about baby", "tags": ["anxiety", "pregnancy"]},
                {"emotion_vector": [0.9, 0.1], "summary": "Had a calm day", "tags": ["joy"]}
            ]

    class MockEmotionCore:
        def get_current_emotion(self):
            return [0.1, 0.9]

    core = CognitivePrioritizationCore(MockMemoryCore(), MockEmotionCore())
    emotion, memories = core.evaluate_current_state()
    ranked = core.rank_memory_relevance(emotion)
    core.generate_thought_pulse(ranked, emotion)
    next_action = core.route_behavior()
    print("Next Action:", next_action)

