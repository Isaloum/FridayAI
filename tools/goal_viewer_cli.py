# =============================================
# File: goal_viewer_cli.py
# Purpose: View, filter, and manage long-term goals from terminal
# =============================================

import json
from pathlib import Path
from tabulate import tabulate

GOALS_PATH = Path("core/goal_data/long_term_goals.json")

if not GOALS_PATH.exists():
    print("[Goal Viewer] No goals found.")
    exit()

with open(GOALS_PATH, 'r', encoding='utf-8') as f:
    goals = json.load(f)

if not goals:
    print("[Goal Viewer] No entries to display.")
    exit()

# === Display Table ===
def display(goals):
    rows = []
    for g in goals:
        rows.append([
            g.get("id"),
            g.get("status"),
            g.get("emotion"),
            g.get("description")[:60] + ("..." if len(g.get("description")) > 60 else ""),
            ", ".join(g.get("tags", [])),
            g.get("origin")
        ])

    headers = ["ID", "Status", "Emotion", "Description", "Tags", "Origin"]
    print("\n🧭 Friday's Long-Term Goals:")
    print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))

# === CLI Prompt ===
def cli():
    display(goals)
    print("\n🔧 Actions: [done <id>] [archive <id>] [delete <id>] [exit]")
    while True:
        cmd = input("Command > ").strip().lower()
        if cmd == "exit":
            break
        elif cmd.startswith("done "):
            gid = cmd.split()[1]
            for g in goals:
                if g.get("id") == gid:
                    g["status"] = "done"
                    print(f"✅ Marked {gid} as done.")
        elif cmd.startswith("archive "):
            gid = cmd.split()[1]
            for g in goals:
                if g.get("id") == gid:
                    g["status"] = "archived"
                    print(f"📦 Archived {gid}.")
        elif cmd.startswith("delete "):
            gid = cmd.split()[1]
            goals[:] = [g for g in goals if g.get("id") != gid]
            print(f"❌ Deleted {gid}.")
        else:
            print("⚠️ Invalid command.")
        display(goals)

    # Save changes
    with open(GOALS_PATH, 'w', encoding='utf-8') as f:
        json.dump(goals, f, indent=2)
    print("💾 Changes saved.")

if __name__ == '__main__':
    cli()
