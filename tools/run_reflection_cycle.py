from core.MemoryCore import MemoryCore
from core.EmotionCoreV2 import EmotionCoreV2
from core.reflective_cognition.ReflectionLoopManager import ReflectionLoopManager

if __name__ == "__main__":
    memory = MemoryCore()
    emotion = EmotionCoreV2()

    reflection_loop = ReflectionLoopManager(memory, emotion)
    reflection_loop.run_reflection_cycle()
