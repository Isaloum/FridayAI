# =============================================
# File: AgentPlanner.py
# Purpose: Parse user goals or prompts into executable tool commands
# =============================================

from ToolExecutor import ToolExecutor

class AgentPlanner:
    """
    Parses natural user requests and selects the appropriate tool + arguments.
    This version uses simple keyword + pattern rules.
    In later stages, this could be upgraded with local LLMs.
    """

    def __init__(self):
        self.executor = ToolExecutor()

    def plan_and_execute(self, user_prompt: str) -> dict:
        """
        Plan which tool to use from a natural input, then execute it.
        Returns result dict.
        """
        lowered = user_prompt.lower()

        # Match: clock / time / now
        if any(k in lowered for k in ["time", "clock", "what time is it"]):
            return self.executor.execute("current_time")

        # Match: math / calculate / solve
        elif any(k in lowered for k in ["calculate", "solve", "math", "equation"]):
            equation = self._extract_equation(user_prompt)
            return self.executor.execute("calculate", equation)

        # Match: shell / run / terminal
        elif any(k in lowered for k in ["run", "terminal", "command", "shell"]):
            cmd = self._extract_command(user_prompt)
            return self.executor.execute("run_shell", cmd)

        return {"success": False, "error": "No matching tool found for input."}

    def _extract_equation(self, prompt):
        # crude fallback – in future, parse via LLM
        return prompt.split("calculate")[-1].strip().strip("? ")

    def _extract_command(self, prompt):
        return prompt.split("run")[-1].strip().strip("? ")


# ====================
# CLI Test Mode
# ====================
if __name__ == "__main__":
    agent = AgentPlanner()
    print("\n🧠 AgentPlanner Test")
    while True:
        try:
            query = input("You: ")
            if query.lower() in ["exit", "quit"]:
                break
            result = agent.plan_and_execute(query)
            print("\n🛠️ Result:", result)
        except KeyboardInterrupt:
            break
