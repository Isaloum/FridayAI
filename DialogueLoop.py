# DialogueLoop.py
# -------------------
# Live chat loop for FridayAI with full brain integration
# Includes memory context injection for smarter, personal replies

from MemoryCore import MemoryCore
from GraphReasoner import GraphReasoner
from GraphBrainCore import GraphBrainCore
from EmotionCore import EmotionCore
from EmpathyCore import EmpathyCore
from WebSearchBrainV2 import WebSearchBrainV2
from IntentRouter import IntentRouter
from MemoryContextInjector import MemoryContextInjector

if __name__ == "__main__":
    print("🤖 FridayAI is online. Type 'exit' to shut down.\n")

    # Initialize brain modules
    memory = MemoryCore()
    graph = GraphBrainCore()
    reasoner = GraphReasoner(graph)
    router = IntentRouter(memory, reasoner)
    memory_injector = MemoryContextInjector()

    while True:
        user_input = input("🗣️ You: ")
        if user_input.lower().strip() in ["exit", "quit"]:
            print("👋 Friday: Talk soon. I'm always here.")
            break

        # Inject past context to enrich the current input
        enriched_input = memory_injector.inject_context(user_input)

        # Route to Friday's brain
        reply = router.route(enriched_input)

        print("🤖 Friday:", reply)
        print("=" * 60)
