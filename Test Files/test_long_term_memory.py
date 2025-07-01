# test_long_term_memory.py
# -------------------------
# Basic unit test for LongTermMemory class

from LongTermMemory import LongTermMemory

def run_tests():
    memory = LongTermMemory("test_memory.json")

    print("🔍 Storing a sample conversation...")
    memory.store("What is the capital of France?", "The capital of France is Paris.")

    print("🔍 Searching for keyword 'France'...")
    results = memory.search("France")
    for res in results:
        print(f"✅ Found: You said '{res['user_input']}', Friday said '{res['reply']}'")

    print("🧾 Summarizing recent conversations...")
    print(memory.summarize_recent(3))

    # Cleanup
    import os
    os.remove("test_memory.json")
    print("🧹 Temporary test file removed.")

if __name__ == "__main__":
    run_tests()
