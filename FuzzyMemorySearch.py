# =====================================
# FuzzyMemorySearch.py - Semantic Key/Value Search
# =====================================

from difflib import SequenceMatcher
from typing import List, Tuple, Optional

class FuzzyMemorySearch:
    def __init__(self, memory_core):
        self.memory = memory_core

    def search(self, query: str, limit: int = 3) -> List[Tuple[str, float]]:
        results = []
        query = query.lower()

        for key in self.memory.memory:
            for version in self.memory.memory[key]:
                value = version['value'].lower()
                score = self._similarity(query, value)
                if score > 0.4:
                    results.append((version['value'], score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def _similarity(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

# =====================
# Example Manual Usage
# =====================
if __name__ == "__main__":
    from MemoryCore import MemoryCore
    memory = MemoryCore()
    searcher = FuzzyMemorySearch(memory)

    while True:
        query = input("Search: ").strip()
        if query.lower() in ("exit", "quit"):
            break

        results = searcher.search(query)
        if not results:
            print("No close matches found.")
        else:
            for text, score in results:
                print(f"\n🔍 Match ({round(score, 2)}): {text}")
