# ======================================
# task_executor.py – Option B: Emotion-Prioritized Executor
# ======================================
#
# This upgraded version adds:
# - Emotional priority sorting
# - Injection into Friday's memory
# - Task tagging with execution timestamp and intensity
# - Post-task reflection via TaskReflectionAgent

import os
import json
from pathlib import Path
from datetime import datetime
import logging
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from core.MemoryCore import MemoryCore
#from TaskReflectionAgent import TaskReflectionAgent
from core.TaskReflectionAgent import TaskReflectionAgent


# === Paths ===
QUEUE_PATH = Path("core/knowledge_data/plans/task_queue.json")
ARCHIVE_PATH = Path("core/knowledge_data/plans/task_archive.json")
LOG_PATH = Path("core/knowledge_data/plans/task_execution.log")
reflection_agent = TaskReflectionAgent()

# === Setup Logging ===
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# === Emotion Weight Table ===
EMOTION_WEIGHTS = {
    "anxiety": 9,
    "bleeding": 10,
    "depression": 8,
    "stress": 7,
    "danger": 8,
    "pain": 6,
    "fear": 7,
    "emergency": 10,
    "risk": 5,
    "urgent": 6,
    "severe": 7,
    "unsafe": 6
}

# === Load Task Queue ===
def load_tasks():
    if not QUEUE_PATH.exists():
        return []
    with open(QUEUE_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

# === Save Archive ===
def archive_tasks(completed):
    ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if ARCHIVE_PATH.exists():
        with open(ARCHIVE_PATH, 'r', encoding='utf-8') as f:
            archive = json.load(f)
    else:
        archive = []
    archive.extend(completed)
    with open(ARCHIVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(archive, f, indent=2)

# === Clear Queue ===
def clear_queue():
    with open(QUEUE_PATH, 'w', encoding='utf-8') as f:
        json.dump([], f)

# === Prioritize Tasks ===
def prioritize(tasks):
    return sorted(tasks, key=lambda t: EMOTION_WEIGHTS.get(t.get("emotion"), 1), reverse=True)

# === Inject into Memory ===
def inject_to_memory(task):
    mem = MemoryCore()
    summary = f"Task: {task['task']} from {task['source']}"
    mem.save_memory({
        "type": "task_execution",
        "content": summary,
        "tags": [task['emotion'], task['source']],
        "source": "TaskExecutor",
        "timestamp": datetime.now().isoformat(),
        "metadata": {
            "executed": True
        }
    })


# === Process ===
def process_tasks():
    tasks = load_tasks()
    if not tasks:
        print("[INFO] No tasks to execute.")
        return

    print(f"[RUNNING] Executing {len(tasks)} prioritized task(s)...")
    tasks = prioritize(tasks)
    completed = []

    for task in tasks:
        summary = f"TASK: {task['task']} (emotion: {task['emotion']}, source: {task['source']})"
        print(summary)
        logging.info(f"EXECUTED: {summary}")

        task["executed_at"] = datetime.now().isoformat()
        task["score"] = EMOTION_WEIGHTS.get(task['emotion'], 1)

        inject_to_memory(task)

        # === Task Reflection Integration ===
        task_result = {
            "task_id": task.get("id", "unknown"),
            "description": task.get("task"),
            "result": f"Executed task from {task['source']}",
            "timestamp": task["executed_at"],
            "emotion": task.get("emotion", "neutral")
        }
        reflection_agent.run_reflection_cycle(task_result)

        completed.append(task)

    archive_tasks(completed)
    clear_queue()
    print(f"[DONE] Executed and archived {len(completed)} task(s).")


if __name__ == '__main__':
    process_tasks()
