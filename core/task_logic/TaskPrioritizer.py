# File: core/task_logic/TaskPrioritizer.py

import json
import os

MEMORY_FILE = "friday_memory.json"

# === UTILS ===

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# === SCORING ENGINE ===

def score_task(task):
    """
    Assigns a score to each task for prioritization.
    Higher = more urgent.
    - Emotion: fear/urgency = +10
    - Tag: 'priority' = +5
    - Tag: 'deadline' = +3
    """
    score = 0
    emotion = task.get("emotion", "").lower()
    tags = task.get("tags", [])

    if emotion in ["urgency", "fear", "anxiety"]:
        score += 10
    if "priority" in tags:
        score += 5
    if "deadline" in tags:
        score += 3

    return score

# === MAIN PRIORITY FUNCTION ===

def get_prioritized_tasks():
    """
    Returns a sorted list of unexecuted tasks by descending priority.
    """
    memory = load_memory()
    tasks = [m for m in memory if m.get("type") == "task" and not m.get("executed", False)]

    # Attach score
    for task in tasks:
        task["_score"] = score_task(task)

    # Sort descending
    sorted_tasks = sorted(tasks, key=lambda t: t["_score"], reverse=True)
    return sorted_tasks
