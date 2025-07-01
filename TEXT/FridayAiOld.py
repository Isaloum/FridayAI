# =====================================
# FridayAI.py
# Main Core of Friday AI Assistant (Smart Modular Version)
# =====================================

import pyttsx3
import speech_recognition as sr
from openai import OpenAI
from MemoryCore import MemoryCore
from NLUProcessor import NLUProcessor
import time

class FridayAI:
    """Main AI class managing input, memory, understanding, and output."""

    def __init__(self, memory_core=None):
        if memory_core:
            self.memory = memory_core
        else:
            from MemoryCore import MemoryCore
            self.memory = MemoryCore()
            
        self.nlu = NLUProcessor(self.memory)
        self.client = OpenAI(api_key="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")  # <<< Insert your real API Key here
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.last_response = ""

    def speak(self, text):
        """Convert text to audio speech."""
        self.engine.say(text)
        self.engine.runAndWait()
        
    def respond_to(self, question):
        """Smart universal response based on available facts."""
        question = question.lower()
        
        # Try to fuzzy match the requested fact
        requested_fact = None
        
        if "favorite" in question or "what is my" in question:
            requested_fact = question.replace("what is my", "").replace("?", "").strip()
        
        if "where am i" in question:
            requested_fact = "location"
        
        if requested_fact:
            fact_value = self.memory.get_fact(requested_fact)
            if fact_value:
                return f"Your {requested_fact} is {fact_value}."
            else:
                return f"Fact '{requested_fact}' not found. Please update or specify."
        
        if "system status" in question:
            return "Systems operational."

        return f"Fact '{question}' not found. Please update or specify."

    def listen(self):
        """Capture audio input from the microphone."""
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("[🎤 Listening...]")
            audio = recognizer.listen(source)
        try:
            command = recognizer.recognize_google(audio)
            print(f"You: {command}")
            return command
        except sr.UnknownValueError:
            print("[⚠️] Sorry, I didn't catch that.")
            return ""
        except sr.RequestError:
            print("[⚠️] Speech recognition service unavailable.")
            return ""

    def chat_with_openai(self, prompt):
        """Fallback to OpenAI GPT-3.5 Turbo with memory context."""
        try:
            context = self.memory.get_context()
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"Context: {context}\nYou are Friday, a helpful intelligent assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[OpenAI Error] {str(e)}"

    def understand(self, user_input):
        """Process input using NLU first, then fallback to GPT."""
        self.memory.add_context(f"User: {user_input}")

        # Run through NLU Processor
        nlu_response = self.nlu.parse(user_input)
        if nlu_response:
            self.memory.add_context(f"Friday: {nlu_response}")
            return nlu_response

        # Fallback: GPT-3.5 turbo if NLU doesn't recognize
        ai_response = self.chat_with_openai(user_input)
        self.memory.add_context(f"Friday: {ai_response}")
        return ai_response

    def run(self):
        """Main loop for Friday operation."""
        print("[FRIDAY] Core Systems Online.")
        print("[FRIDAY] Awaiting your commands...")

        while True:
            try:
                user_input = input("You: ")

                if user_input.lower() in ["exit", "quit", "shutdown"]:
                    response = "Acknowledged. Shutting down systems. Goodbye Boss!"
                    print(f"Friday: {response}")
                    self.speak(response)
                    break

                if user_input == "":
                    continue

                response = self.understand(user_input)
                print(f"Friday: {response}")
                self.last_response = response

                # Speak only if user types "speak"
                if user_input.lower() == "speak":
                    self.speak(self.last_response)

            except Exception as e:
                print(f"[FRIDAY ERROR] {e}")
                time.sleep(2)
                continue

# =====================================
# Launch Friday
# =====================================
if __name__ == "__main__":
    assistant = FridayAI()
    assistant.run()
