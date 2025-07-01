import pyautogui
import requests
import os
import time

os.makedirs("Book_Assets/Screenshots", exist_ok=True)

while True:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"Book_Assets/Screenshots/{timestamp}.png"
    pyautogui.screenshot(filename)
    print(f"Saved: {filename}")
    time.sleep(900)