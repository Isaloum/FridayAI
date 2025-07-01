# ======================================
# File: InterpreterBridge.py
# Purpose: Interface with Open Interpreter for code and command execution.
# ======================================

try:
    from open_interpreter import interpreter
except ImportError:
    print("⚠️ Open Interpreter not installed.")
    interpreter = None

class InterpreterBridge:
    def run(self, command: str) -> str:
        if interpreter:
            return interpreter.run(command)
        return "Interpreter not available."

    def run_code(self, code: str) -> str:
        if interpreter:
            return interpreter.run_code(code)
        return "Interpreter not available."
