# File: core/task_logic/TaskExecutor.py

import json
import os
from datetime import datetime

# === CONFIGURATION ===
MEMORY_FILE = "friday_memory.json"

# === MEMORY UTILITIES ===

def load_memory():
    """Load all memory entries from disk."""
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_memory(memory):
    """Write all memory entries back to disk."""
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)

def append_memory(entry):
    """Append a new memory entry and save."""
    memory = load_memory()
    memory.append(entry)
    save_memory(memory)

# === TASK EXECUTION ===

def execute_task(task):
    """
    Simulate task execution.
    In future, this could trigger real actions. For now, it's a placeholder.
    Returns a dict describing the outcome.
    """
    # Simulate "doing" the task
    description = task.get("description", "unknown task")
    outcome_text = f"Simulated execution of task: '{description}'"

    # Return execution result
    return {
        "status": "success",
        "details": outcome_text
    }

def run_task_execution(verbose=False):
    """
    Find and execute all unexecuted tasks from memory.
    Logs the result and updates task status.
    """
    memory = load_memory()
    executed_count = 0

    for entry in memory:
        if entry.get("type") == "task" and not entry.get("executed", False):
            result = execute_task(entry)

            # Update task
            entry["executed"] = True
            entry["outcome"] = result["status"]
            entry["executed_at"] = datetime.utcnow().isoformat()

            # Create reflection memory
            from core.reflective_cognition.ExecutionFeedback import build_reflection

            reflection = build_reflection(entry, result["status"], result["details"])

            append_memory(reflection)

            if verbose:
                print(f"[✓] Executed: {entry['description']}")
                print(f"    → {result['details']}")

            executed_count += 1

    save_memory(memory)

    if verbose:
        print(f"✅ {executed_count} task(s) executed.")
    return executed_count
