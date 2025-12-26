class MemoryCore:
    def __init__(self):
        self.history = []

    def add_context(self, role, content=None):
        # This handles cases where only one argument or two are sent
        if content is None:
            # If only one thing was sent, treat it as the content
            actual_content = role
            actual_role = "user"
        else:
            actual_role = role
            actual_content = content
            
        self.history.append({"role": actual_role, "content": actual_content})
        print(f"[MEMORY] Logged: {actual_content}")

    def get_context(self):
        return self.history
