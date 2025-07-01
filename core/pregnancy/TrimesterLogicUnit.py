# ==============================================
# File: core/pregnancy/TrimesterLogicUnit.py
# Purpose: Determine trimester stage from week input
# ==============================================

class TrimesterLogicUnit:
    @staticmethod
    def get_trimester(weeks: int) -> str:
        if weeks < 0:
            raise ValueError("Weeks must be a non-negative integer.")
        if weeks < 13:
            return "first"
        elif weeks < 28:
            return "second"
        else:
            return "third"


# ==========================
# CLI Test Mode
# ==========================
if __name__ == "__main__":
    while True:
        try:
            weeks = int(input("Enter weeks pregnant (or -1 to quit): "))
            if weeks == -1:
                break
            trimester = TrimesterLogicUnit.get_trimester(weeks)
            print(f"🍼 Trimester: {trimester}")
        except Exception as e:
            print("❌ Error:", e)
