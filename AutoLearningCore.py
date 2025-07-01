# AutoLearningCore.py - Centralized Fact Learning Engine

import re
import uuid
from datetime import datetime
from typing import List, Dict

class AutoLearningCore:
    def __init__(self, memory_core, graph_core=None):
        self.memory = memory_core
        self.graph = graph_core  # optional, future link to GraphBrain

    def process_input(self, user_input: str, reply: str = None) -> List[Dict]:
        """
        Extracts and saves learnable facts from input. Returns list of learned facts.
        """
        learned = []
        ts = datetime.now().isoformat()

        # Rule: Name extraction
        match = re.search(r"my name is ([A-Z][a-z]+)", user_input, re.I)
        if match:
            name = match.group(1).strip()
            self.memory.save_fact("user.name", name, source="auto_learned", metadata={"timestamp": ts})
            learned.append({"key": "user.name", "value": name})

        # Rule: Project or task
        match = re.search(r"(i\'m|i am|i have been) (building|working on|creating) ([A-Za-z0-9_-]+)", user_input, re.I)
        if match:
            project = match.group(3).strip()
            self.memory.save_fact("project.current", project, source="auto_learned", metadata={"timestamp": ts})
            learned.append({"key": "project.current", "value": project})

        # Rule: Emotional statement
        if any(emotion in user_input.lower() for emotion in ["i feel", "i'm feeling", "i was feeling"]):
            self.memory.save_fact("emotion.statement", user_input, source="auto_learned", metadata={"timestamp": ts})
            learned.append({"key": "emotion.statement", "value": user_input})

        # 🔁 Also store raw input as general memory with tag and importance scoring
        self.learn_from_text(user_input, source="auto_learned")

        return learned

    def _generate_tags(self, content):
        tags = []
        topics = {
            'AI': ['artificial intelligence', 'machine learning', 'neural network', 'consciousness', 'quantum computing'],
            'health': ['pain', 'injury', 'therapy', 'medication'],
            'tech': ['elon musk', 'spacex', 'tesla', 'robot', 'software'],
            'emotion': ['happy', 'sad', 'angry', 'love', 'depressed']
        }

        content_lower = content.lower()
        for tag, keywords in topics.items():
            for word in keywords:
                if word in content_lower:
                    tags.append(tag)
                    break
        return tags or ['general']

    def _estimate_importance(self, content):
        if len(content) > 200:
            return 0.9
        elif len(content) > 100:
            return 0.6
        else:
            return 0.3

    def learn_from_text(self, content, source="user"):
        tags = self._generate_tags(content)
        importance = self._estimate_importance(content)
        timestamp = datetime.now().isoformat()
        key = f"auto_learned/{uuid.uuid4().hex[:8]}"

        self.memory.save_fact(
            key=key,
            value=content,
            source=source,
            metadata={
                "tags": tags,
                "importance": importance,
                "timestamp": timestamp
            }
        )

        return {
            "content": content,
            "tags": tags,
            "importance": importance,
            "timestamp": timestamp
        }

    def learn_from_input_output(self, user_input: str, ai_output: str, metadata: dict = None):
        """
        Saves both user input and AI output as linked memory entries.
        """
        timestamp = datetime.now().isoformat()
        base_meta = metadata or {}
        base_meta["timestamp"] = timestamp

        # Save user input as learnable memory
        self.learn_from_text(user_input, source="cli")

        # Save Friday’s response
        self.memory.save_fact(
            key=f"friday.output.{timestamp}",
            value=ai_output,
            source="friday",
            metadata={
                "emotion": base_meta.get("emotion", "unknown"),
                "timestamp": timestamp
            }
        )

# ==========================
# Example usage
# ==========================
if __name__ == "__main__":
    from MemoryCore import MemoryCore
    memory = MemoryCore(memory_file='friday_memory.enc', key_file='memory.key')
    learner = AutoLearningCore(memory)

    inputs = [
        "My name is Ihab",
        "I'm building FridayAI",
        "I feel overwhelmed by this project"
    ]

    for text in inputs:
        print(f"\nProcessing: {text}")
        results = learner.process_input(text)
        print("Learned:", results)

    print("\n--- Summary ---")
    print(memory.summarize_user_data())
