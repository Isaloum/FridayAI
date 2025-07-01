"""
WebSearchBrainV2.py
----------------------
FridayAI's upgraded web search brain using SerpAPI.
This version returns real-time search results with smart fallbacks.
"""

from serpapi.google_search import GoogleSearch
from dotenv import load_dotenv
import os

load_dotenv()  # Load SERPAPI_KEY from .env

class WebSearchBrainV2:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("SERPAPI_KEY")

    def lookup(self, query: str, max_results=3) -> dict:
        if not self.api_key:
            return {"source": "SerpAPI", "answer": "No API key provided."}

        try:
            search = GoogleSearch({
                "q": query,
                "api_key": self.api_key,
                "num": max_results
            })
            results = search.get_dict()

            if "organic_results" in results:
                top = results["organic_results"][0]
                return {
                    "source": "Google (SerpAPI)",
                    "answer": top.get("snippet", "No snippet available."),
                    "url": top.get("link", "No link available."),
                    "title": top.get("title", "No title")
                }
        except Exception as e:
            return {"source": "SerpAPI", "answer": f"Search error: {str(e)}"}

        return {"source": "SerpAPI", "answer": "No results found."}

# Test directly
if __name__ == "__main__":
    ws = WebSearchBrainV2()
    queries = [
        "top car brands 2024",
        "how many calories in 1 egg",
        "latest AI breakthroughs",
    ]

    for q in queries:
        print(f"\n🔍 Query: {q}")
        result = ws.lookup(q)
        print(f"🌐 Source: {result.get('source', 'Unknown')}")
        print(f"🧠 Title: {result.get('title', 'No title')}")
        print(f"📄 Answer: {result.get('answer', 'No answer')}")
        if result.get("url"):
            print(f"🔗 URL: {result['url']}")
        else:
            print("🔗 URL: (not available)")
        print("\n" + "="*60 + "\n")

