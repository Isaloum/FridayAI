from __future__ import annotations
import os

def write_code_to_file(filename, code):
    """Allows the Mind to write or update its own code files."""
    try:
        with open(filename, 'w') as f:
            f.write(code)
        return f"✅ Successfully wrote code to {filename}"
    except Exception as e:
        return f"❌ Failed to write file: {e}"

def read_local_file(filename):
    """Allows the Mind to read any file in the current directory."""
    try:
        with open(filename, 'r') as f:
            return f.read()
    except Exception as e:
        return f"❌ Failed to read file: {e}"
