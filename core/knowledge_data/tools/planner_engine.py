# ======================================
# planner_engine.py – Autonomous Goal & Task Generation
# ======================================
#
# This module gives Friday the ability to create and manage internal goals,
# based on memory review, emotion trends, and domain focus.
#
# CMD Usage:
# python planner_engine.py
#
# Dependencies: datetime, json, random, pathlib

import json
import random
from datetime import datetime
from pathlib import Path

# Files
PLANNER_FILE = Path("core/knowledge_data/plans/task_queue.json")
PROFILE_FILE = Path("core/knowledge_data/profile_manager.json")

# Default task templates per domain
TASK_TEMPLATES = {
    "pregnancy": [
        "Scrape updates on prenatal nutrition",
        "Summarize emotional logs for last 48h",
        "Search for new treatments for gestational diabetes",
        "Generate weekly wellness checklist"
    ],
    "mechanic": [
        "Scrape technical specs for fuel systems",
        "Summarize troubleshooting cases",
        "Embed latest oil compatibility charts",
        "Generate diagnostic check routine"
    ],
    "law": [
        "Scrape new articles on contract disputes",
        "Summarize rulings on AI liability",
        "Update legal memory index",
        "Draft outline on case analysis method"
    ]
}


def load_active_domain():
    """Pull current active domain from profile manager."""
    with open(PROFILE_FILE, 'r', encoding='utf-8') as f:
        profile = json.load(f)
    return profile.get("current_domain", "pregnancy")


def generate_tasks(domain: str):
    """Randomly create 2–3 tasks from known template pool."""
    pool = TASK_TEMPLATES.get(domain, [])
    return random.sample(pool, min(3, len(pool)))


def save_tasks(tasks):
    """Save generated tasks to the planner queue file."""
    PLANNER_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "timestamp": datetime.utcnow().isoformat(),
        "tasks": tasks
    }
    with open(PLANNER_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"[PLANNER] Tasks generated and saved to {PLANNER_FILE}")


if __name__ == '__main__':
    domain = load_active_domain()
    tasks = generate_tasks(domain)
    save_tasks(tasks)

    print("\n[PLANNER OUTPUT]\n-------------------------")
    for t in tasks:
        print(f"✅ {t}")
