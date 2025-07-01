# File: tools/run_execution_cycle.py

import time
import sys
import os

# Add task logic to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../core/task_logic")))

from TaskExecutor import run_task_execution

# === CONFIGURATION ===
CYCLE_INTERVAL_SECONDS = 10  # How often to check for tasks (can be changed)

def main():
    print("🔁 Starting FridayAI Task Execution Loop")
    print(f"⏱️  Checking for tasks every {CYCLE_INTERVAL_SECONDS} seconds...\n")

    try:
        while True:
            count = run_task_execution(verbose=True)
            if count == 0:
                print("🟡 No new tasks found.")
            else:
                print(f"✅ Executed {count} task(s).")
            print("Sleeping...\n")
            time.sleep(CYCLE_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\n🛑 Execution loop stopped by user.")

if __name__ == "__main__":
    main()
