# =====================================
# File: tools/InterpreterBridge.py
# Purpose: Connects FridayAI to Open Interpreter's local execution engine
# =====================================

import subprocess
import tempfile
import os

class InterpreterBridge:
    """
    This module allows Friday to execute local shell commands or Python code
    through the Open Interpreter engine (interpreter.exe or similar).
    """

    def __init__(self, executable_path: str = "interpreter.exe"):
        self.executable = executable_path
        if not os.path.exists(self.executable):
            raise FileNotFoundError(f"Interpreter binary not found: {self.executable}")

    def run(self, command: str) -> str:
        """Run a shell or generic command."""
        try:
            result = subprocess.run(
                [self.executable, "--command", command],
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.stdout.strip() or result.stderr.strip()
        except Exception as e:
            return f"⚠️ Execution error: {str(e)}"

    def run_code(self, python_code: str) -> str:
        """Run Python code using interpreter engine."""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix=".py", delete=False) as tmp:
                tmp.write(python_code)
                tmp_path = tmp.name

            result = subprocess.run(
                [self.executable, "--file", tmp_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            os.remove(tmp_path)
            return result.stdout.strip() or result.stderr.strip()

        except Exception as e:
            return f"⚠️ Python execution error: {str(e)}"

    def read_file(self, filepath: str) -> str:
        """Utility: Read a file's contents from disk."""
        if not os.path.isfile(filepath):
            return f"⚠️ File not found: {filepath}"
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"⚠️ Failed to read file: {str(e)}"
