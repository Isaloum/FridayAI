"""
GenerativeResponder.py
-----------------------
Generates a text response using OpenAI or falls back to Ollama if no API key is set.

Usage (CMD):
    python GenerativeResponder.py --prompt "Hello, Friday."

Requirements:
    - OPENAI_API_KEY (environment variable) for OpenAI usage
    - Ollama server must be running at http://localhost:11434 for fallback
"""

import os
import json
import argparse
import requests
import openai

# Use OpenAI if API key is available
def call_openai(prompt):
    try:
        # Create OpenAI client using API key from environment variable
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Send prompt to GPT-4
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )

        # Return response content + metadata
        return {"response": response.choices[0].message.content, "source": "openai"}

    except Exception as e:
        # Return error if anything goes wrong
        return {"error": str(e), "source": "openai"}

# Use Ollama as local fallback
def call_ollama(prompt):
    try:
        # Send prompt to local mistral model via Ollama API
        payload = {
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        }

        res = requests.post("http://localhost:11434/api/generate", json=payload)

        # Return formatted response if success
        if res.status_code == 200:
            result = res.json()
            return {"response": result.get("response", "").strip(), "source": "ollama"}
        else:
            return {"error": f"HTTP {res.status_code}", "source": "ollama"}

    except Exception as e:
        # Return error if request fails
        return {"error": str(e), "source": "ollama"}

# Route the request based on environment
def route_llm(prompt):
    if os.getenv("OPENAI_API_KEY"):
        return call_openai(prompt)
    else:
        return call_ollama(prompt)

# Command line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to send to LLM")
    args = parser.parse_args()

    # Get and print response
    result = route_llm(args.prompt)
    print(json.dumps(result, indent=2))
