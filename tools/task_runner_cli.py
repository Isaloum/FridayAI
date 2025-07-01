# File: tools/task_runner_cli.py

import argparse
import sys
import os

# Add path to import TaskExecutor
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../core/task_logic")))

from TaskExecutor import run_task_execution

# === CLI TOOL ===

def main():
    parser = argparse.ArgumentParser(description="FridayAI Task Runner CLI")
    parser.add_argument("--run-once", action="store_true", help="Run pending tasks once and exit.")
    parser.add_argument("--verbose", action="store_true", help="Print detailed task logs.")

    args = parser.parse_args()

    if args.run_once:
        print("🚀 Executing pending tasks...")
        count = run_task_execution(verbose=args.verbose)
        if count == 0:
            print("🟡 No unexecuted tasks found.")
        else:
            print(f"✅ {count} task(s) executed.")
    else:
        print("⚠️ No mode selected. Use --run-once to run the task processor.")

if __name__ == "__main__":
    main()
