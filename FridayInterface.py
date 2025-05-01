# FridayInterface.py
from FridayAI import FridayAI

def main():
    print("Friday AI Activated. Type 'exit' to quit.")
    friday = FridayAI()
    
    try:
        while True:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit']:
                print("Friday: Goodbye! Have a great day!")
                break

            # Process the query
            response = friday.respond_to(user_input)
            
            # Handle response
            if response.get('status') == 'error':
                print(f"Friday: ⚠️ Error {response.get('error_code', '')} - {response.get('content', 'Unknown error')}")
            else:
                print(f"Friday: {response.get('content', 'Hmm, let me think about that...')}")
                friday.speak(response.get('content', ''))
                
    except KeyboardInterrupt:
        print("\nFriday: Session terminated unexpectedly")

if __name__ == "__main__":
    main()