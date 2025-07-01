# =============================================
# File: FieldRegistry.py
# Purpose: Dynamically register and activate FridayAI domains by field
# =============================================

class FieldRegistry:
    def __init__(self):
        self.domains = {}
        self.active_domains = []

    def register(self, domain_name: str, modules: list):
        """
        Register a field domain with its list of callable modules
        """
        self.domains[domain_name.lower()] = modules

    def activate(self, domain_name: str):
        """
        Activate a specific domain for use by FridayAI
        """
        domain = domain_name.lower()
        if domain in self.domains and domain not in self.active_domains:
            self.active_domains.append(domain)
            print(f"[Registry] Activated domain: {domain}")
        elif domain not in self.domains:
            print(f"[Registry] Domain not found: {domain}")

    def get_active_modules(self):
        """
        Return a flat list of all callable modules in active domains
        """
        active_modules = []
        for domain in self.active_domains:
            active_modules.extend(self.domains.get(domain, []))
        return active_modules

    def list_available_domains(self):
        return list(self.domains.keys())

    def reset(self):
        self.active_domains = []
        print("[Registry] Domain registry reset.")


# ================== Example Setup ==================

if __name__ == "__main__":
    def mock_module():
        return "Hello from a domain module."

    registry = FieldRegistry()

    registry.register("health", [mock_module])
    registry.register("engineering", [mock_module])
    registry.register("law", [mock_module])

    registry.activate("health")
    registry.activate("engineering")

    print("Active Modules:", registry.get_active_modules())
    print("Available Domains:", registry.list_available_domains())
