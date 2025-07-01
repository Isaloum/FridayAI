
from KnowledgeCore import KnowledgeCore
from MemoryCore import MemoryCore
from rapidfuzz import fuzz

class KnowledgeRouter:
    """
    Central routing logic for FridayAI.
    Routes queries based on known domains and integrates memory + knowledge core.
    """

    def __init__(self):
        self.knowledge_core = KnowledgeCore()
        self.memory = MemoryCore()
        self.domain_keywords = {
            "transport": ["transport", "bus", "train", "taxi", "uber", "car", "airport"],
            "emergency": ["emergency", "911", "help", "hospital", "police"],
        }

    def detect_domain(self, query: str) -> str:
        query_lower = query.lower()
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for kw in keywords if fuzz.partial_ratio(kw, query_lower) > 80)
            if score >= 2:
                return domain
        return "general"

    def handle_query(self, query: str) -> dict:
        # Try known knowledge first
        response = self.knowledge_core.lookup(query)
        if response:
            return {
                "domain": "knowledge",
                "content": response,
                "confidence": 0.9
            }

        # Fallback to memory facts
        fact = self.memory.get_fact(query)
        if fact:
            return {
                "domain": "memory",
                "content": f"Previously you told me: {fact['value']}",
                "confidence": 0.8
            }

        return {
            "domain": "unknown",
            "content": "I don't have an answer for that yet, but I can learn if you teach me.",
            "confidence": 0.2
        }

    def add_fact(self, label: str, response: str, keywords=None):
        """
        Adds new knowledge to the live core.
        :param label: A unique key string
        :param response: The response text to return
        :param keywords: A list of relevant keywords for fuzzy matching
        """
        if not keywords:
            keywords = label.lower().split()
        self.knowledge_core.knowledge_db[label] = {
            "response": response,
            "keywords": set(keywords)
        }
