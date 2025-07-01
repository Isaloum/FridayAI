import os
import pyautogui
import time
from datetime import datetime

# Corrected path to FridayAI
FOLDER = os.path.join(os.environ['USERPROFILE'], "FridayAI", "exports", "screenshots")

def main():
    os.makedirs(FOLDER, exist_ok=True)  # Auto-creates folders
    print(f"📂 Screenshots will save to:\n{FOLDER}\n")
    
    try:
        while True:
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            path = os.path.join(FOLDER, f"{now}.png")
            pyautogui.screenshot(path)
            print(f"✅ Saved: {path}")
            time.sleep(900)  # 15 minutes
            
    except KeyboardInterrupt:
        print("\n🚫 Stopped by user")
    except Exception as e:
        print(f"🔥 CRASHED: {str(e)}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()