# =====================================
# test_planning_core.py – CMD Planner Test
# =====================================

from MemoryCore import MemoryCore
from PlanningCore import PlanningCore

# Initialize memory
memory = MemoryCore(memory_file="test_memory.enc", key_file="test.key")

# Inject data with repeated emotional patterns
memory.save_fact("note1", "I’m exhausted from constant pain", metadata={"tags": ["pain"], "emotion": {"tired": 1}})
memory.save_fact("note2", "Pain ruined my sleep again", metadata={"tags": ["pain", "sleep"], "emotion": {"tired": 1}})
memory.save_fact("note3", "Pain made it hard to work", metadata={"tags": ["pain", "work"], "emotion": {"frustrated": 1}})
memory.save_fact("note4", "I can’t focus at work", metadata={"tags": ["work", "focus"], "emotion": {"anxious": 1}})
memory.save_fact("note5", "Pain is always there when I try to sleep", metadata={"tags": ["pain", "sleep"], "emotion": {"tired": 1}})

# Run planner
planner = PlanningCore(memory)
print("\n📅 Suggested Plans:\n")
print(planner.suggest_plan())

# Clean up
import os
os.remove("test_memory.enc")
os.remove("test.key")
