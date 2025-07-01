# GPT4Core.py
import openai
import os

class GPT4Core:
    def __init__(self, api_key=None, model="gpt-4"):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model
        openai.api_base = "https://openrouter.ai/api/v1"
        openai.api_key = self.api_key

    def prompt(self, user_prompt: str, system_prompt: str = "You are Friday, a helpful, emotionally-aware AI.") -> str:
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[GPT4 Error]: {e}"
