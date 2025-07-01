# =====================================
# GraphBrainCore.py - Concept Linkage System
# =====================================
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import re

class GraphBrainCore:
    """Builds and manages conceptual links across memory using co-occurrence and topic proximity."""

    def __init__(self, memory_core):
        self.memory = memory_core
        self.graph = defaultdict(Counter)
        self._build_graph()

    def _extract_tags_from_entry(self, entry: Dict) -> List[str]:
        """Extracts tags from an entry’s metadata or inferred from text content."""
        tags = entry.get("metadata", {}).get("tags", [])
        if not tags:
            text = str(entry.get("value", "")).lower()
            tags = re.findall(r'\b[a-z]{4,}\b', text)
        return list(set(tags))

    def _build_graph(self):
        """Constructs the concept graph from memory."""
        entries = self.memory.get_all()
        for entry in entries:
            tags = self._extract_tags_from_entry(entry)
            for i in range(len(tags)):
                for j in range(i + 1, len(tags)):
                    t1, t2 = sorted([tags[i], tags[j]])
                    self.graph[t1][t2] += 1
                    self.graph[t2][t1] += 1

    def find_related_concepts(self, topic: str, top_n: int = 5) -> List[Tuple[str, int]]:
        """Returns top related concepts connected to the given topic."""
        topic = topic.lower()
        if topic not in self.graph:
            return []
        return self.graph[topic].most_common(top_n)

    def explain_links_naturally(self, topic: str) -> str:
        """Returns a conversational explanation of related concepts."""
        related = self.find_related_concepts(topic)
        if not related:
            return f"I haven't seen much around the topic '{topic}' yet."

        output = f"When you talk about '{topic}', you often also bring up:\n"
        for concept, strength in related:
            output += f"– {concept}\n"
        return output.strip()
