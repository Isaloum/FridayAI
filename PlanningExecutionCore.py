# ======================================
# File: PlanningExecutionCore.py
# Purpose: Handle user goals, break them into steps, track execution, and generate summary reflections.
# ======================================

import re
from typing import Optional, Dict, Any, List
from tools.InterpreterBridge import InterpreterBridge
from datetime import datetime

class PlanningExecutionCore:
    def __init__(self):
        self.master_plan = []  # List of plans with structure: {goal, steps, completed, summary, timestamps}
        self.current_step = None

    def detect_plan_intent(self, user_input: str) -> Optional[str]:
        # Filter out very short or overly casual inputs to avoid misfires (e.g., "heyyyyy", "hola")
        if len(user_input.strip()) < 8 or re.match(r'^(hi+|hey+|yo+|hola+)$', user_input.lower()):
            return None

        # Soft check for semantic planning-related terms
        planning_keywords = ["plan", "goal", "task", "mission", "schedule", "step", "next", "milestone"]
        if any(word in user_input.lower() for word in planning_keywords):
            return "planning_intent_detected"

        return None

    def update_or_create_plan(self, user_input: str) -> str:
        goal = self._extract_goal(user_input)
        if not goal:
            goal = "default_plan"

        timestamp = datetime.now().isoformat()
        existing_plan = next((p for p in self.master_plan if p['goal'] == goal), None)

        if existing_plan:
            existing_plan['summary'] = user_input
            existing_plan['last_updated'] = timestamp
            return f"🔁 Updated plan: **{goal}** — noted: '{user_input}'"
        else:
            plan = {
                "goal": goal,
                "steps": [],
                "completed": [],
                "summary": user_input,
                "created": timestamp,
                "last_updated": timestamp
            }
            self.master_plan.append(plan)
            return f"🧠 New plan created for '{goal}'."

    def create_structured_plan(self, goal: str, steps: List[str]) -> str:
        timestamp = datetime.now().isoformat()
        plan = {
            "goal": goal,
            "steps": steps,
            "completed": [],
            "summary": f"Structured plan with {len(steps)} steps.",
            "created": timestamp,
            "last_updated": timestamp
        }
        self.master_plan.append(plan)
        return f"🧠 Structured plan for '{goal}' created with {len(steps)} steps."
    
    # === [LLM AUTO STEP PARSER] ===
    def create_structured_plan_from_text(self, goal_text: str) -> str:
        from tools.LLMPlannerBridge import LLMPlannerBridge
        llm = LLMPlannerBridge()
        steps = llm.generate_steps(goal_text)
        
        timestamp = datetime.now().isoformat()
        plan = {
            "goal": goal_text,
            "steps": steps,
            "completed": [],
            "summary": f"LLM generated {len(steps)} step(s) for: '{goal_text}'",
            "created": timestamp,
            "last_updated": timestamp
        }
        self.master_plan.append(plan)
        return f"🧠 Auto-plan for '{goal_text}' created with {len(steps)} steps."

    def next_step(self) -> str:
        for plan in self.master_plan:
            for step in plan["steps"]:
                if step not in plan["completed"]:
                    self.current_step = step
                    return f"🔄 Next step for '{plan['goal']}': {step}"
        return "✅ All steps completed."

    def complete_step(self) -> str:
        if not self.current_step:
            return "⚠️ No active step."

        for plan in self.master_plan:
            if self.current_step in plan["steps"] and self.current_step not in plan["completed"]:
                plan["completed"].append(self.current_step)
                done = self.current_step
                self.current_step = None
                plan['last_updated'] = datetime.now().isoformat()
                return f"✅ Step completed: {done}"

        return "⚠️ Step not recognized in any plan."

    def get_active_plan_context(self) -> Dict[str, Any]:
        if not self.master_plan:
            return {"summary": "No active plans."}
        latest = sorted(self.master_plan, key=lambda x: x['last_updated'], reverse=True)[0]
        return {"summary": f"Latest plan: '{latest['summary']}' (Last updated: {latest['last_updated']})"}

    def reflect_on_plan_progress(self) -> str:
        if not self.master_plan:
            return "There are no active plans at the moment."
        return "\n".join([f"🗂️ {p['goal']}: {len(p['completed'])}/{len(p['steps'])} steps completed ({int((len(p['completed'])/len(p['steps']) * 100) if p['steps'] else 0)}%)"
                          for p in self.master_plan])

    def plan_summary(self) -> str:
        if not self.master_plan:
            return "🕳 No plans available."
        summary = []
        for plan in self.master_plan:
            goal = plan['goal']
            done = len(plan['completed'])
            total = len(plan['steps'])
            percent = int((done / total) * 100) if total > 0 else 0
            summary.append(f"📋 {goal}: {done}/{total} steps completed ({percent}%)")
        return "\n".join(summary)

    def delete_plan(self, goal: str) -> str:
        before = len(self.master_plan)
        self.master_plan = [p for p in self.master_plan if p['goal'] != goal]
        after = len(self.master_plan)
        return f"🗑 Plan for '{goal}' removed." if before != after else "⚠️ No such plan found."

    def _extract_goal(self, user_input: str) -> Optional[str]:
        # Attempt to extract a meaningful goal from the user's sentence
        match = re.search(r"(plan|goal|task)\s+(to\s+)?(.+)", user_input.lower())
        if match:
            return match.group(3).strip()
        return None
        

    def execute_step(self) -> str:
        bridge = InterpreterBridge()
        
        for plan in self.master_plan:
            for step in plan["steps"]:
                if step not in plan["completed"]:
                    self.current_step = step
                    result = ""

                    if isinstance(step, dict):
                        if step.get("type") == "shell":
                            result = bridge.run(step["command"])
                        elif step.get("type") == "python":
                            result = bridge.run_code(step["code"])
                        else:
                            result = f"⚠️ Unknown step type: {step.get('type')}"
                    else:
                        # Legacy support for plain string steps
                        result = bridge.run(step)

                    plan["completed"].append(step)
                    plan["last_updated"] = datetime.now().isoformat()
                    return f"🔧 Executed step: {step}\n🧾 Result:\n{result}"

        return "✅ All steps have already been executed."
