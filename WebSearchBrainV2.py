"""
WebSearchBrainV2.py
----------------------
FridayAI's upgraded web search brain using SerpAPI.
This version returns real-time search results with smart fallbacks.
"""

from serpapi import GoogleSearch
import os
from dotenv import load_dotenv
load_dotenv()

class WebSearchBrainV2:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("SERPAPI_KEY")

    def lookup(self, query: str, max_results=3) -> dict:
        """
        Use SerpAPI to perform a real-time Google search.
        Returns top result with title, snippet, and URL.
        """
        if not self.api_key:
            return {"source": "SerpAPI", "answer": "No API key provided."}

        try:
            search = GoogleSearch({
                "q": query,
                "api_key": self.api_key,
                "num": max_results
            })
            results = search.get_dict()

            if "organic_results" in results and results["organic_results"]:
                top = results["organic_results"][0]
                return {
                    "source": "Google (SerpAPI)",
                    "answer": top.get("snippet", "No snippet available."),
                    "url": top.get("link"),
                    "title": top.get("title", "No title")
                }
        except Exception as e:
            return {"source": "SerpAPI", "answer": f"Search error: {str(e)}"}

        return {"source": "SerpAPI", "answer": "No results found."}

# --------------------
# Test it standalone
# --------------------
if __name__ == "__main__":
    import getpass
    print("🔐 Enter SerpAPI Key:")
    key = getpass.getpass()
    ws = WebSearchBrainV2(api_key=key)

    queries = [
        "who is the president of Canada",
        "Elon Musk 2025",
        "how many grams of protein in 1 egg",
        "latest news about AI"
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
