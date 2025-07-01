
# test_query_memory.py
# --------------------
# CMD test file for QueryMemoryCore with encrypted MemoryCore and proper tag injection using normalized keys

from QueryMemoryCore import QueryMemoryCore
from MemoryCore import MemoryCore
import os

# ✅ Use test-specific encrypted memory so you don't corrupt real data
test_config = {
    'memory_file': 'test_query.enc',
    'key_file': 'test_query.key'
}

# Cleanup: Delete test files if they already exist
for f in [test_config['memory_file'], test_config['key_file']]:
    if os.path.exists(f):
        os.remove(f)

# ✅ Initialize encrypted MemoryCore and inject 3 memory entries
memory = MemoryCore(**test_config)

# Helper to get normalized key
def get_normalized_key(memory_core, raw_key):
    return memory_core._normalize_key(raw_key)

# Save entries and inject tags using normalized keys
memory.save_fact("cnesst.case", "Talked about CNESST and chronic pain in left shoulder.", "test", metadata={"emotion": {"sad": 1}})
key = get_normalized_key(memory, "cnesst.case")
memory.memory[key][-1]["tags"] = ["cnesst", "pain", "injury"]

memory.save_fact("doctor.report", "Feeling anxious about the doctor's report.", "test", metadata={"emotion": {"anxious": 1}})
key = get_normalized_key(memory, "doctor.report")
memory.memory[key][-1]["tags"] = ["doctor", "report", "anxious"]

memory.save_fact("gym.stress", "Workout helped reduce my stress today.", "test", metadata={"emotion": {"happy": 1}})
key = get_normalized_key(memory, "gym.stress")
memory.memory[key][-1]["tags"] = ["gym", "stress", "happy"]

# ✅ Initialize QueryMemoryCore
query = QueryMemoryCore(memory)

# 🔍 Test 1: Summarize by tag
print("\n🔍 Searching memories with tag 'cnesst':")
print(query.summarize(tag="cnesst"))

# 🧠 Test 2: Reflect emotional trends
print("\n🧠 Reflecting emotional trends from the last 14 days:")
print(query.reflect_emotions(days=14))

# 📌 Test 3: Frequent tags/topics
print("\n📌 Frequent tags/topics across memory:")
print(query.get_frequent_topics())

# 🔎 Test 4: Natural language search
print("\n🔎 Querying memory for: 'CNESST pain doctor report':")
results = query.query_memory("CNESST pain doctor report", days=90)
for i, entry in enumerate(results, 1):
    print(f"\nResult {i}: [{entry['timestamp']}] {entry['value']}")
