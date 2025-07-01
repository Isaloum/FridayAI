# File: tools/execution_logger_cli.py

import json
import os
from rich import print
from datetime import datetime

MEMORY_FILE = "friday_memory.json"

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        print("[red]❌ Memory file not found.[/red]")
        return []
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def show_executed_tasks():
    memory = load_memory()
    executed_tasks = [m for m in memory if m.get("type") == "task" and m.get("executed")]

    if not executed_tasks:
        print("[yellow]⚠️ No executed tasks found.[/yellow]")
        return

    print(f"[bold green]📘 Executed Tasks Log ({len(executed_tasks)} total):[/bold green]\n")

    for idx, task in enumerate(executed_tasks, 1):
        print(f"[cyan]#{idx}[/cyan] [bold]{task['description']}[/bold]")
        print(f"    [blue]Time:[/blue] {task.get('executed_at', 'Unknown')}")
        print(f"    [blue]Emotion:[/blue] {task.get('emotion', 'N/A')}")
        print(f"    [blue]Tags:[/blue] {', '.join(task.get('tags', []))}")
        print(f"    [blue]Result:[/blue] {task.get('execution_result', 'n/a')}")
        print("")

if __name__ == "__main__":
    print("[bold]🧠 FridayAI Execution Log Viewer[/bold]")
    print("=====================================\n")
    show_executed_tasks()
