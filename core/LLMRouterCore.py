"""
LLMRouterCore.py
----------------
This module routes prompts to either:
- OpenAI GPT-4 (if OPENAI_API_KEY is set)
- Ollama local model (if not)

Usage:
    from core.LLMRouterCore import route_llm
    result = route_llm("What's your purpose?")
    print(result["response"])
"""

import os
import requests
import openai

# Main router function
def route_llm(prompt: str) -> dict:
    # 🔐 Use OpenAI if API key is set
    if os.getenv("OPENAI_API_KEY"):
        try:
            # Create OpenAI client (v1.x style)
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            # Send prompt to GPT-4
            res = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )

            # Return text response + source
            return {"response": res.choices[0].message.content, "source": "openai"}

        except Exception as e:
            return {"error": str(e), "source": "openai"}

    # 🧠 Otherwise use Ollama
    else:
        try:
            # Send prompt to local mistral model
            res = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "mistral", "prompt": prompt, "stream": False}
            )

            # Parse and return Ollama response
            out = res.json() if res.status_code == 200 else {}
            return {"response": out.get("response", "").strip(), "source": "ollama"}

        except Exception as e:
            return {"error": str(e), "source": "ollama"}
