# =============================================
# File: narrative_viewer_cli.py
# Purpose: View Friday's self-narrative from the terminal
# =============================================

import json
from pathlib import Path
from datetime import datetime

NARRATIVE_PATH = Path("core/memory_bank/self_narrative_log.json")

def load_narrative():
    if not NARRATIVE_PATH.exists():
        print("[Narrative] No narrative log found.")
        return []
    with open(NARRATIVE_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def display_narrative(entries):
    print("\n🧠 Friday's Self-Narrative:")
    print("=" * 40)
    for entry in sorted(entries, key=lambda x: x['timestamp']):
        timestamp = entry.get("timestamp", "unknown")
        kind = entry.get("kind", "reflection")
        content = entry.get("content", "[No content]")
        print(f"[{timestamp}] <{kind}>\n{content}\n")

if __name__ == '__main__':
    entries = load_narrative()
    if entries:
        display_narrative(entries)
    else:
        print("[Narrative] No entries to display.")
