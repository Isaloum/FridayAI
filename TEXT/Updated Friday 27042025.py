# Import necessary libraries
import json
import os
import random
from datetime import datetime
import speech_recognition as sr
import pyttsx3
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(api_key="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")  # <--- Replace with your real API key

class Friday:
    def __init__(self, memory_file='friday_memory.json', personal_file='friday_personal_memory.json'):
        self.memory_file = memory_file
        self.personal_file = personal_file
        self.memory = self.load_memory()
        self.personal_memory = self.load_personal_memory()
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)

    def load_memory(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as f:
                return json.load(f)
        return []

    def save_memory(self):
        with open(self.memory_file, 'w') as f:
            json.dump(self.memory, f, indent=4)

    def load_personal_memory(self):
        if os.path.exists(self.personal_file):
            with open(self.personal_file, 'r') as f:
                return json.load(f)
        return {}

    def save_personal_memory(self):
        with open(self.personal_file, 'w') as f:
            json.dump(self.personal_memory, f, indent=4)

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def listen(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source)
        try:
            command = recognizer.recognize_google(audio)
            print(f"You said: {command}")
            return command
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
            return ""
        except sr.RequestError:
            print("Sorry, speech service is down.")
            return ""

    def remember_fact(self, fact_key, fact_value):
        self.personal_memory[fact_key.lower()] = fact_value
        self.save_personal_memory()
        return f"I have updated that your {fact_key} is {fact_value}."

    def recall_fact(self, fact_key):
        fact = self.personal_memory.get(fact_key.lower())
        if fact:
            return f"Your {fact_key} is {fact}."
        else:
            return f"I do not have that information yet."

    def save_conversation_entry(self, role, message):
        conversation_file = 'friday_conversation_memory.json'
        conversation = []
        if os.path.exists(conversation_file):
            try:
                with open(conversation_file, 'r') as f:
                    conversation = json.load(f)
            except:
                conversation = []
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "role": role,
            "message": message
        }
        conversation.append(entry)
        with open(conversation_file, 'w') as f:
            json.dump(conversation, f, indent=4)

    def generate_response(self, user_input):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are Friday, an intelligent, loyal personal assistant."},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.5,
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error contacting GPT: {e}"

    def interview_user(self):
        if not self.personal_memory:
            print("Starting first-time setup to personalize your assistant...")
            questions = [
                ("name", "What is your name?"),
                ("favorite color", "What is your favorite color?"),
                ("favorite car", "What is your favorite car?"),
                ("hobby", "What is your favorite hobby?"),
                ("goal", "What is your biggest goal in life?")
            ]
            for key, question in questions:
                answer = input(f"{question} ")
                if answer:
                    self.remember_fact(key, answer)

    def chat(self, user_input):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if user_input.lower().startswith("remember "):
            try:
                key_value = user_input[9:].split(" is ")
                key = key_value[0].strip()
                value = key_value[1].strip()
                response = self.remember_fact(key, value)
                self.memory.append({"timestamp": timestamp, "user": user_input, "friday": response})
                self.save_memory()
                return response
            except:
                return "Please use the format: remember [thing] is [value]."
                
        if user_input.lower().startswith("update "):
            try:
                key_value = user_input[7:].split(" to ")
                key = key_value[0].strip()
                value = key_value[1].strip()
                response = friday.remember_fact(key, value)
                self.memory.append({"timestamp": timestamp, "user": user_input, "friday": response})
                self.save_memory()
                return response
            except:
        return "Please use the format: update [thing] to [new value]."

        if user_input.lower() == "review memory":
            return self.review_memory()

        if user_input.lower() == "delete memory":
            self.memory.clear()
            self.save_memory()
            return "Memory has been wiped clean."

        if user_input.lower().startswith("search "):
            keyword = user_input[7:].strip()
            return self.search_memory(keyword)

        response = self.generate_response(user_input)
        self.memory.append({"timestamp": timestamp, "user": user_input, "friday": response})
        self.save_memory()
        return response

    def review_memory(self):
        if not self.memory:
            return "No prior conversations stored."
        review = "\n".join([
            f"[{entry['timestamp']}] You: {entry['user']} | Friday: {entry['friday']}"
            for entry in self.memory
        ])
        return review

    def search_memory(self, keyword):
        results = [
            f"[{entry['timestamp']}] You: {entry['user']} | Friday: {entry['friday']}"
            for entry in self.memory
            if keyword.lower() in entry['user'].lower() or keyword.lower() in entry['friday'].lower()
        ]
        if not results:
            return "No matching entries found."
        return "\n".join(results)

# Main system
if __name__ == "__main__":
    friday = Friday()
    print("[FRIDAY] Online. How can I assist you?")

    friday.interview_user()

    while True:
        try:
            user_input = input("You: ")

            if user_input.lower() in ["exit", "quit", "shutdown"]:
                response = "Acknowledged. Executing shutdown sequence."
                print(f"Friday: {response}")
                friday.speak(response)
                break

            if user_input == "":
                continue

            response = friday.chat(user_input)
            print(f"Friday: {response}")
            last_response = response

            # Save conversation turn
            friday.save_conversation_entry("user", user_input)
            friday.save_conversation_entry("friday", response)

            # Full Natural Auto-Memory Extraction
            try:
                analysis_prompt = f"Extract any personal facts from this sentence: '{user_input}'. Format it as JSON with key-value pairs. If none, return empty JSON."
                analysis_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a fact extractor. Only respond in JSON format like {\"favorite color\": \"blue\"}."},
                        {"role": "user", "content": analysis_prompt}
                    ],
                    temperature=0.0,
                    max_tokens=100
                )
                facts_json = analysis_response.choices[0].message.content.strip()

                if facts_json and facts_json.startswith("{") and facts_json.endswith("}"):
                    facts = json.loads(facts_json)
                    for key, value in facts.items():
                        friday.remember_fact(key, value)
                        print(f"Friday: Noted — {key} is {value}.")

            except Exception as e:
                print(f"[Memory Extraction Error] {e}")

            if user_input.lower() == "speak":
                friday.speak(last_response)
                continue

        except Exception as e:
            print(f"[Error] {e}")
            print("[FRIDAY] Attempting safe recovery...")
            continue
