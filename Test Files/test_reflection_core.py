# =====================================
# test_reflection_core.py - CMD Reflection Test
# =====================================

from MemoryCore import MemoryCore
from ReflectionCore import ReflectionCore

# Initialize
memory = MemoryCore(memory_file="test_memory.enc", key_file="test.key")

# Inject pattern-based memory
memory.save_fact("note1", "My back pain is killing me", metadata={"tags": ["pain", "back"], "emotion": {"frustrated": 1}})
memory.save_fact("note2", "I can't sleep because of this", metadata={"tags": ["sleep", "pain"], "emotion": {"tired": 1}})
memory.save_fact("note3", "Work stress is nonstop", metadata={"tags": ["work", "stress"], "emotion": {"overwhelmed": 1}})
memory.save_fact("note4", "Even the gym hurts my back", metadata={"tags": ["pain", "gym"], "emotion": {"frustrated": 1}})
memory.save_fact("note5", "Pain makes it hard to focus", metadata={"tags": ["pain", "focus"], "emotion": {"frustrated": 1}})

# Test
reflector = ReflectionCore(memory)
print("\n🪞 Reflection Report:\n")
print(reflector.reflect_on_patterns(days=7))

# Cleanup
import os
os.remove("test_memory.enc")
os.remove("test.key")
