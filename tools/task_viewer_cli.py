# =============================================
# File: task_viewer_cli.py
# Purpose: Simple CLI to show saved tasks from memory
# =============================================

from core.MemoryCore import MemoryCore
import json

def main():
    memory = MemoryCore()
    all_entries = memory.query_memories(since_hours=72)
    
    # Only show entries marked as "task"
    tasks = [m for m in all_entries if m.get("type") == "task"]

    print(f"[TaskViewer] Found {len(tasks)} task(s):\n")
    for task in tasks:
        print(json.dumps(task, indent=2))
        print("-" * 60)

if __name__ == "__main__":
    main()
