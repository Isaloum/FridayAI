# =====================================
# test_dialogue.py - FridayAI Dialogue Testing
# =====================================

from MemoryCore import MemoryCore
from EmotionCore import EmotionCore  # If you have it
from DialogueCore import DialogueCore

# Initialize cores
memory = MemoryCore(memory_file='test_memory.enc', key_file='test.key')
emotion = EmotionCore() if 'EmotionCore' in globals() else None  # Optional
dialogue = DialogueCore(memory, emotion)

# Add test memory data
memory.save_fact("test.note.1", "I had a rough day at work", metadata={"emotion": {"sad": 1}})
memory.save_fact("test.note.2", "I went to the gym and felt better", metadata={"emotion": {"happy": 1}})
memory.save_fact("test.note.3", "Finished building FridayAI core memory system", metadata={"emotion": {"proud": 1}})
memory.save_fact("note4", "Pain keeps affecting my sleep", metadata={"tags": ["pain", "sleep"], "emotion": {"tired": 1}})
memory.save_fact("note5", "I hate how pain slows me down", metadata={"tags": ["pain"], "emotion": {"frustrated": 1}})
memory.save_fact("note6", "Pain really made it hard to sleep again", metadata={"tags": ["pain", "sleep"], "emotion": {"tired": 1}})

# Test 1: Memory Summary
response = dialogue.respond_to("What do you remember about me?")
print("\n🧠 Test 1: Memory Summary\n", response['content'])

# Test 2: Emotion (if EmotionCore implemented)
if emotion:
    response = dialogue.respond_to("How am I feeling?")
    print("\n💬 Test 2: Emotion Reflection\n", response['content'])

# Test 3: ReflectionCore Response
response = dialogue.respond_to("Is there anything I keep bringing up?")
print("\n🪞 Test 3: ReflectionCore Response\n", response['content'])
response = dialogue.respond_to("Do you have any suggestions for me?")
print("\n📅 Test 4: PlanningCore Response\n", response['content'])

# Clean up test files
import os
os.remove("test_memory.enc")
os.remove("test.key")


