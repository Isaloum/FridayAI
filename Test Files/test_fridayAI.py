# =====================================
# test_fridayAI.py - Modular Test Scaffold
# =====================================

import os
from datetime import datetime
from MemoryCore import MemoryCore
from DialogueCore import DialogueCore

# Optional: add when EmotionCore is implemented
try:
    from EmotionCore import EmotionCore
    emotion = EmotionCore()
except ImportError:
    emotion = None

# -------------------------------------
# CONFIG
# -------------------------------------
MEMORY_FILE = "test_memory.enc"
KEY_FILE = "test.key"

# Clean any prior test files
if os.path.exists(MEMORY_FILE):
    os.remove(MEMORY_FILE)
if os.path.exists(KEY_FILE):
    os.remove(KEY_FILE)

# Initialize
memory = MemoryCore(memory_file=MEMORY_FILE, key_file=KEY_FILE)
dialogue = DialogueCore(memory, emotion)

# -------------------------------------
# TEST SETUP
# -------------------------------------
print("\n🔧 Seeding test memory...")
memory.save_fact("journal.day1", "Today I felt anxious and couldn't focus", metadata={"emotion": {"anxious": 1}})
memory.save_fact("journal.day2", "Had a great workout and felt strong", metadata={"emotion": {"happy": 1}})
memory.save_fact("project.friday", "Finished implementing core memory system", metadata={"emotion": {"proud": 1}})
print("✅ Test entries added.")

# -------------------------------------
# TEST CASES
# -------------------------------------

# Test 1: Memory Summary
print("\n🧠 Test 1: Memory Summary Trigger")
response = dialogue.respond_to("What do you remember about me?")
print(response["content"])

# Test 2: Emotion Reflection (if EmotionCore available)
if emotion:
    print("\n💬 Test 2: Emotion Analysis")
    response = dialogue.respond_to("How am I feeling?")
    print(response["content"])
else:
    print("\nℹ️ Skipping Emotion Test (EmotionCore not available)")

# -------------------------------------
# CLEANUP
# -------------------------------------
os.remove(MEMORY_FILE)
os.remove(KEY_FILE)
print("\n🧹 Test memory cleared.")
