# =============================================
# File: ToolRegistry.py
# Purpose: Register and expose callable tool functions
# =============================================

from typing import Callable, Dict
import subprocess
import datetime
import math

class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self._register_builtin_tools()

    def _register_builtin_tools(self):
        self.register("current_time", lambda: datetime.datetime.now().isoformat())
        self.register("run_shell", self.run_shell)
        self.register("calculate", self.calculate_expression)

    def register(self, name: str, func: Callable):
        self.tools[name] = func

    def get_tool(self, name: str) -> Callable:
        return self.tools.get(name)

    def list_tools(self):
        return list(self.tools.keys())

    def run(self, name: str, *args, **kwargs):
        tool = self.get_tool(name)
        if tool:
            return tool(*args, **kwargs)
        return f"❌ Tool '{name}' not found."

    # === Tool Functions ===

    def run_shell(self, command: str) -> str:
        try:
            result = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, timeout=10)
            return result.decode("utf-8").strip()
        except Exception as e:
            return f"Shell error: {e}"

    def calculate_expression(self, expression: str) -> str:
        try:
            return str(eval(expression, {"__builtins__": {}}, math.__dict__))
        except Exception as e:
            return f"Math error: {e}"


# ====================
# CLI Test Mode
# ====================
if __name__ == "__main__":
    registry = ToolRegistry()
    print("🛠️ Available Tools:", registry.list_tools())

    print("\n⏱️ Time:", registry.run("current_time"))
    print("\n🧮 Math:", registry.run("calculate", "sqrt(49) + log(10)"))
    print("\n💻 Shell:", registry.run("run_shell", "echo Hello Friday"))
