# =============================================
# File: GoalManager.py
# Purpose: Manage Friday's evolving long-term goals
# =============================================

import json
from pathlib import Path
from datetime import datetime

GOALS_PATH = Path("core/goal_data/long_term_goals.json")

class GoalManager:
    def __init__(self):
        self.goals = []
        self.path = GOALS_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump([], f)

    def load_goals(self):
        with open(self.path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_goals(self, goals):
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(goals, f, indent=2)

    def add_goal(self, description, tags=None, emotion="neutral", origin="manual", extra_fields=None):
        if tags is None:
            tags = []

        goal_id = f"goal_{len(self.goals) + 1:03}"
        goal = {
            "id": goal_id,
            "description": description,
            "tags": tags,
            "emotion": emotion,
            "status": "active",
            "origin": origin,
            "converted_to_task": False,
            "created_at": datetime.now().isoformat()
        }

        # If extra fields are passed, merge them in
        if extra_fields:
            goal.update(extra_fields)

        self.goals.append(goal)
        self.save_goals(self.goals)
        print(f"[GoalManager] Added goal: {goal_id}")


    def update_goal_status(self, goal_id, new_status):
        goals = self.load_goals()
        for goal in goals:
            if goal["id"] == goal_id:
                goal["status"] = new_status
                goal["updated"] = datetime.now().isoformat()
                self.save_goals(goals)
                print(f"[GoalManager] Updated {goal_id} to {new_status}")
                return
        print(f"[GoalManager] Goal not found: {goal_id}")

    def get_active_goals(self):
        return [g for g in self.load_goals() if g["status"] == "active"]

    def archive_goal(self, goal_id):
        self.update_goal_status(goal_id, "archived")
