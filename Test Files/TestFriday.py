# TestFriday.py (Updated Version)
from FridayAI import FridayAI

def main():
    friday = FridayAI()
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        print("Friday:", friday.respond_to(user_input))

if __name__ == "__main__":
    main()