import os
from dotenv import load_dotenv
from MemoryCore import MemoryCore
from NLUProcessor import NLUProcessor
import groq

load_dotenv()

class Friday:
    def __init__(self):
        self.memory = MemoryCore()
        self.nlu = NLUProcessor()
        self.client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
        print("[FRIDAY] Core Systems Online.")

    def run(self):
        print("[FRIDAY] Awaiting your commands...")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]: break
            
            self.memory.add_context("user", user_input)
            
            # Use Groq to get a real response
            try:
                completion = self.client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": user_input}]
                )
                response = completion.choices[0].message.content
                print(f"Friday: {response}")
                self.memory.add_context("assistant", response)
            except Exception as e:
                print(f"[ERROR] Brain connection failed: {e}")

if __name__ == "__main__":
    bot = Friday()
    bot.run()
