# =====================================
# MemoryInspector.py - Debugging & Visualization Tool
# =====================================

from typing import List
from rich import print
from rich.table import Table

class MemoryInspector:
    def __init__(self, memory):
        self.memory = memory

    def list_all_keys(self) -> List[str]:
        return list(self.memory.memory.keys())

    def show_latest(self, limit: int = 5):
        table = Table(title="🧠 FridayAI Memory Snapshot (Latest)")
        table.add_column("Key")
        table.add_column("Value")
        table.add_column("Timestamp")

        keys = list(self.memory.memory.keys())[-limit:][::-1]
        for key in keys:
            fact = self.memory.memory[key][-1]
            val = fact['value']
            time = fact['timestamp']
            table.add_row(key, val, time)

        print(table)

    def show_fact(self, key: str):
        from pprint import pprint
        norm_key = self.memory._normalize_key(key)
        if norm_key not in self.memory.memory:
            print("[red]No such memory key.[/red]")
            return

        print(f"\n[bold]🔍 Full memory for key: {key}[/bold]\n")
        for i, version in enumerate(self.memory.memory[norm_key]):
            print(f"[green]Version {i+1}[/green]:")
            pprint(version)
            print("---")

    def run_interactive(self):
        print("\n[bold cyan]🧠 FridayAI Memory Inspector Interface[/bold cyan]\n")
        while True:
            cmd = input("Inspector > ").strip().lower()
            if cmd in ("exit", "quit"):
                break
            elif cmd == "list":
                print(self.list_all_keys())
            elif cmd == "latest":
                self.show_latest()
            elif cmd.startswith("view "):
                key = cmd.replace("view ", "", 1)
                self.show_fact(key)
            else:
                print("[yellow]Unknown command. Try: list, latest, view <key>, exit[/yellow]")

# =====================
# Example Manual Usage
# =====================
if __name__ == "__main__":
    from MemoryCore import MemoryCore
    mem = MemoryCore()
    inspector = MemoryInspector(mem)
    inspector.run_interactive()
