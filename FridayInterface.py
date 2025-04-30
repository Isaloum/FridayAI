# FridayInterface.py
from FridayAI import FridayAI

def main():
    friday = FridayAI()
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit']:
                print("Friday: Goodbye!")
                break
                
            response = friday.respond_to(user_input)

            if isinstance(response, dict) and response.get("requires_followup"):
                data_type = response["requires_followup"]
                user_data = input(f"Friday: {response['response']}\nYou: ")
                friday.memory.save_fact(f"user_{data_type}", user_data)
                print(f"Friday: Thanks! I'll remember your {data_type}.")
            else:
                output = response if isinstance(response, str) else response.get('response', '...')
                print(f"Friday: {output}")
                friday.speak(output)
                
        except KeyboardInterrupt:
            print("\nFriday: Goodbye!")
            break

if __name__ == "__main__":
    main()