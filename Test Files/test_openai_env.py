import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

try:
    result = openai.Completion.create(
        engine="text-davinci-003",
        prompt="Say hello to Jarvis.",
        max_tokens=10
    )
    print("✅ API Key is working:")
    print(result.choices[0].text.strip())

except Exception as e:
    print("❌ API key failed:", str(e))
