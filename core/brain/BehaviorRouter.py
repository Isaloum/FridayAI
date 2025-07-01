# ==============================================
# File: core/brain/BehaviorRouter.py
# Purpose: Route next action to correct domain
# ==============================================

class BehaviorRouter:
    def __init__(self, domain_registry):
        self.domains = domain_registry  # dict like {"pregnancy": PregnancySupportCore()}

    def route(self, target):
        if target == "mental_health":
            if "pregnancy" in self.domains:
                self.domains["pregnancy"].soothe_anxiety()
        elif target == "journal":
            if "pregnancy" in self.domains:
                self.domains["pregnancy"].log_emotion()
        else:
            print("No route taken. Target:", target)

if __name__ == "__main__":
    class MockPregnancySupport:
        def soothe_anxiety(self):
            print("[Pregnancy] → soothing anxiety")

        def log_emotion(self):
            print("[Pregnancy] → logging journal entry")

    registry = {"pregnancy": MockPregnancySupport()}
    router = BehaviorRouter(registry)
    router.route("mental_health")
    router.route("journal")
    router.route("unknown")
