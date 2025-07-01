# =============================================
# File: ToolExecutor.py
# Purpose: Interface between planner and tool registry, handles execution + error capture
# =============================================

from ToolRegistry import ToolRegistry

class ToolExecutor:
    def __init__(self):
        self.registry = ToolRegistry()

    def execute(self, tool_name: str, *args, **kwargs) -> dict:
        """
        Attempt to run a tool and return structured result.
        """
        try:
            result = self.registry.run(tool_name, *args, **kwargs)
            return {
                "success": True,
                "tool": tool_name,
                "result": result
            }
        except Exception as e:
            return {
                "success": False,
                "tool": tool_name,
                "error": str(e)
            }

    def available_tools(self):
        return self.registry.list_tools()


# ====================
# CLI Test Mode
# ====================
if __name__ == "__main__":
    executor = ToolExecutor()
    print("\n🧪 ToolExecutor Test")
    print("📦 Tools:", executor.available_tools())

    print("\n⏱️ Time Result:", executor.execute("current_time"))
    print("\n🧮 Calc Result:", executor.execute("calculate", "sin(1) + cos(1)"))
    print("\n💻 Shell Result:", executor.execute("run_shell", "echo Test Success"))
