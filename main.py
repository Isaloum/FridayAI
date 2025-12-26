from __future__ import annotations
import os
from dotenv import load_dotenv
from graph_logic import FridayGraph

load_dotenv()

def run_mind():
    # Initialize the Graph Nervous System
    friday_engine = FridayGraph().build()
    state = {"messages": [], "current_task": ""}

    print("🧠 FridayAI Mind: 'Graph Logic Engaged. I am ready for complex tasks.'")
    
    while True:
        user_input = input("\n👤 Command: ")
        if user_input.lower() in ["exit", "quit"]: break
        
        # Add user message to state
        state["messages"].append({"role": "user", "content": user_input})
        
        # Run the Graph
        result = friday_engine.invoke(state)
        state = result # Update memory
        
        print(f"\n🧠 RESPONSE:\n{result['messages'][-1].content}")

if __name__ == "__main__":
    run_mind()
