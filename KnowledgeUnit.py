from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict

@dataclass
class KnowledgeUnit:
    source: str                    # Where it came from: 'user', 'system', 'plugin', etc.
    content: str                   # The actual information
    timestamp: datetime            # When it was learned
    tags: List[str] = field(default_factory=list)   # Topics, categories
    importance: float = 0.5        # 0 to 1 scale
    metadata: Optional[Dict] = field(default_factory=dict)  # Any extra info (e.g., emotion, topic ID)


class MemoryCore:
    def __init__(self):
        self._memory: List[KnowledgeUnit] = []

    def store(self, content: str, source: str = 'user', tags: Optional[List[str]] = None,
              importance: float = 0.5, metadata: Optional[Dict] = None):
        unit = KnowledgeUnit(
            source=source,
            content=content,
            timestamp=datetime.now(),
            tags=tags or [],
            importance=importance,
            metadata=metadata or {}
        )
        self._memory.append(unit)
        return unit

    def get_all(self) -> List[KnowledgeUnit]:
        return self._memory

    def get_fact(self, key: str) -> Optional[str]:
        # Simplified lookup: just match key in content for now
        for unit in reversed(self._memory):
            if key.lower() in unit.content.lower():
                return unit.content
        return None

    def search_by_tag(self, tag: str) -> List[KnowledgeUnit]:
        return [unit for unit in self._memory if tag in unit.tags]

    def recent(self, limit: int = 5) -> List[KnowledgeUnit]:
        return sorted(self._memory, key=lambda x: x.timestamp, reverse=True)[:limit]
