# ==============================================
# File: core/FridayInterface.py
# Purpose: CLI entry and callable interface to FridayAI
# ==============================================

class FridayInterface:
    def __init__(self):
       #from FridayAI import FridayAI  # Lazy import to prevent circularity
        import sys, os
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from FridayAI import FridayAI

        from core.MemoryCore import MemoryCore
        from core.EmotionCoreV2 import EmotionCoreV2

        self.memory = MemoryCore()
        self.emotion = EmotionCoreV2()
        self.friday = FridayAI(self.memory, self.emotion)

    def launch(self):
        print("🧠 Friday AI is ready. Type 'exit' to quit.")

        try:
            while True:
                user_input = input("\nYou: ").strip()
                if not user_input:
                    continue

                if user_input.lower() in ['exit', 'quit']:
                    print("Friday: Goodbye! Have a great day!")
                    break

                response = self.friday.respond_to(user_input)

                if response.get('status') == 'error':
                    print(f"Friday: ⚠️ Error {response.get('error_code', '')} - {response.get('content', 'Unknown error')}")
                else:
                    print(f"Friday: {response.get('content', 'Hmm, let me think about that...')}")
                    if hasattr(self.friday, "speak"):
                        self.friday.speak(response.get('content', ''))

        except KeyboardInterrupt:
            print("\nFriday: Session terminated by user.")

# Optional CLI trigger
if __name__ == "__main__":
    from FridayAI import FridayAI
    interface = FridayInterface()
    interface.launch()
