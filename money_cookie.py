# ==============================
# money_cookie.py
# (Copy ALL of this into a new file)
# ==============================
import re
import json
from datetime import datetime

class CookieMachine:
    """Simple money-making AI tool"""
    
    def __init__(self):
        self.cookie_jars = {}  # Our memory jar
        self.price_list = {
            'basic': 50,
            'super': 100,
            'mega': 200
        }
    
    def bake_cookie(self, text: str) -> dict:
        """Turn words into money cookies"""
        # Look for important words
        money_words = {
            'love': 2,
            'hate': -2,
            'buy': 3,
            'problem': -1
        }
        
        # Count special words
        score = 0
        found_words = []
        for word, value in money_words.items():
            if word in text.lower():
                score += value
                found_words.append(word)
        
        # Make the money report
        return {
            'score': score,
            'words': found_words,
            'price': self._set_price(score),
            'time': datetime.now().strftime("%Y-%m-%d %H:%M")
        }
    
    def _set_price(self, score: int) -> int:
        """Choose cookie price"""
        if score > 3:
            return self.price_list['mega']
        elif score > 0:
            return self.price_list['super']
        else:
            return self.price_list['basic']

# ======================
# Let's Play!
# ======================  
machine = CookieMachine()

print("🎉 Welcome to Money Cookie Machine! 🎉")
print("Paste customer messages to make money!")

while True:
    text = input("\nPaste text (or 'quit'): ")
    
    if text.lower() == 'quit':
        break
        
    cookie = machine.bake_cookie(text)
    
    print(f"\n💵 Money Report 💵")
    print(f"Score: {cookie['score']}")
    print(f"Found: {', '.join(cookie['words'])}")
    print(f"Charge: ${cookie['price']}")
    print(f"Time: {cookie['time']}")

print("\n💰 Great job! Time to make real money! 💰")