# save_as: C:\Users\ihabs\screenshot_capture.py
import os
import pyautogui
import time

def main():
    base_dir = os.path.join(os.environ['USERPROFILE'], "FridayAI_BookAssets")
    screenshot_dir = os.path.join(base_dir, "Screenshots")
    
    try:
        os.makedirs(screenshot_dir, exist_ok=True)
        print(f"Created directory: {screenshot_dir}")
        
        while True:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(screenshot_dir, f"{timestamp}.png")
            pyautogui.screenshot(filename)
            print(f"Saved: {filename}")
            time.sleep(900)
            
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()