from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    project=os.getenv("OPENAI_PROJECT")
)

try:
    response = client.models.list()
    print("✅ OPENAI CONNECTED — You’re authorized.")
    for model in response.data:
        print("•", model.id)
except Exception as e:
    print("❌ ERROR:", e)
