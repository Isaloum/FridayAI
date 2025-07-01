# ======================================
# File: LLMPlannerBridge.py
# Purpose: Use LLM to convert goal into executable steps.
# ======================================

try:
    from openai import OpenAI
    client = OpenAI()
except:
    client = None

class LLMPlannerBridge:
    def __init__(self, model="gpt-4"):
        self.model = model

    def generate_steps(self, goal_text: str):
        if not client:
            return [{"type": "note", "command": "OpenAI client not available."}]
        
        prompt = f"""
You are a Python automation planner. The user has this goal: "{goal_text}"

Break it into executable steps. For each step, return:
- type: either "shell" or "python"
- command/code: the exact command or Python code

Respond ONLY in valid JSON like:
[
  {{"type": "shell", "command": "pip install numpy"}},
  {{"type": "python", "code": "import numpy as np; print(np.arange(5))"}}
]
        """

        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        try:
            return eval(response.choices[0].message.content)
        except:
            return [{"type": "note", "command": "Failed to parse steps from LLM."}]
