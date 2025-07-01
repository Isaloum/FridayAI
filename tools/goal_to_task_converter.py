# File: tools/goal_to_task_converter.py

import json
import os
import argparse
from datetime import datetime

# === CONFIGURATION ===
GOALS_FILE = "core/goal_logic/goals.json"  # Adjust as needed for your system
MEMORY_FILE = "friday_memory.json"         # Long-term memory file path

# === MEMORY UTILITIES ===

def load_goals():
    """Load goals from the goal JSON store."""
    if not os.path.exists(GOALS_FILE):
        return []
    with open(GOALS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_goals(goals):
    """Save the updated goals list."""
    with open(GOALS_FILE, "w", encoding="utf-8") as f:
        json.dump(goals, f, indent=2)

def save_memory(entry):
    """Append a new memory entry to the main memory file."""
    memory_data = []
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            memory_data = json.load(f)
    memory_data.append(entry)
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory_data, f, indent=2)

# === CORE LOGIC ===

def convert_goals_to_tasks(verbose=False):
    """
    Convert unconverted goals into task entries.
    Each task is saved into memory, and the goal is flagged as converted.
    """
    goals = load_goals()
    converted_count = 0

    for goal in goals:
        if not goal.get("converted_to_task", False):
            task = {
                "type": "task",
                "description": goal["description"],
                "emotion": goal.get("emotion", ""),
                "tags": goal.get("tags", []),
                "converted_from_goal": goal["id"],
                "timestamp": datetime.utcnow().isoformat()
            }
            save_memory(task)
            goal["converted_to_task"] = True
            converted_count += 1
            if verbose:
                print(f"[✓] Task created from Goal ID: {goal['id']}")
                print(f"    → {task['description']}")

    save_goals(goals)

    if verbose:
        print(f"✔️ Converted {converted_count} goal(s) into task(s).")
    return converted_count

# === CLI ENTRY POINT ===

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert stored goals into executable tasks.")
    parser.add_argument("--run-once", action="store_true", help="Run the goal-to-task conversion once.")
    parser.add_argument("--verbose", action="store_true", help="Print each task creation.")
    args = parser.parse_args()

    if args.run_once:
        print("🚀 Running Goal-to-Task Converter...")
        count = convert_goals_to_tasks(verbose=args.verbose)
        if count == 0:
            print("🟡 No new goals needed conversion.")
        else:
            print(f"✅ {count} goal(s) successfully converted.")
