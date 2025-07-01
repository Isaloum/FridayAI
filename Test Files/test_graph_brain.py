# =====================================
# test_graph_brain.py - CMD test for GraphBrainCore
# =====================================
from MemoryCore import MemoryCore
from GraphBrainCore import GraphBrainCore

memory = MemoryCore(memory_file='test_memory.enc', key_file='test.key')

# Add related concept examples
memory.save_fact("note1", "My shoulder pain is getting worse", metadata={"tags": ["pain", "shoulder", "health"]})
memory.save_fact("note2", "This project is exhausting", metadata={"tags": ["project", "fatigue", "mental"]})
memory.save_fact("note3", "The accident caused nerve issues", metadata={"tags": ["accident", "pain", "nerve"]})
memory.save_fact("note4", "Health is impacted when I work too hard", metadata={"tags": ["health", "work", "project"]})

brain = GraphBrainCore(memory)

print("\n🧠 What’s linked to 'pain'?")
print(brain.explain_links_naturally("pain"))

print("\n🧠 What’s linked to 'project'?")
print(brain.explain_links_naturally("project"))

# Clean up
import os
os.remove("test_memory.enc")
os.remove("test.key")
