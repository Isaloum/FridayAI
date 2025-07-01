from MemoryCore import MemoryCore
from AutoLearningCore import AutoLearningCore

memory = MemoryCore()
learner = AutoLearningCore(memory, graph_core=None)

# 🧠 Test any sentence — not limited to name/project
inputs = [
    "I watched a documentary about quantum computing and consciousness.",
    "Pain has been worse in my left shoulder after physiotherapy.",
    "I'm planning to deploy FridayAI with a Raspberry Pi.",
    "The AI should be able to infer, link, and reason across topics.",
    "My cat’s name is Zorro and he’s the reason I get up every day."
]

for text in inputs:
    print(f"\n🧠 Processing: {text}")
    results = learner.process_input(text)
    for r in results:
        print("✅ Structured Fact:", r)

print("\n📦 Memory Summary:")
print(memory.summarize_user_data())
