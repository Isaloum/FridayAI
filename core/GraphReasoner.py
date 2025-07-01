"""
GraphReasoner.py
------------------
This module lets FridayAI think over its graph-based brain.
It explores linked concepts and makes simple inferences based on connections.
"""

from collections import deque

class GraphReasoner:
    """
    GraphReasoner traverses the GraphBrainCore to discover relationships,
    draw conclusions, and answer basic inference questions.
    """

    def __init__(self, graph_brain):
        """
        Initialize with a reference to the GraphBrainCore.
        :param graph_brain: instance of GraphBrainCore
        """
        self.graph = graph_brain

    def path_exists(self, concept_a, concept_b, max_depth=4):
        """
        Check if two concepts are related within a certain depth.
        This is like asking: "Are these ideas connected in the brain?"
        :return: Boolean (True if connected)
        """
        visited = set()
        queue = deque([(concept_a, 0)])

        while queue:
            current, depth = queue.popleft()
            if current == concept_b:
                return True
            if depth < max_depth:
                for neighbor in self.graph.get_connections(current):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))
        return False

    def explain_connection(self, concept_a, concept_b):
        """
        Tries to explain how two concepts are connected, step by step.
        :return: List of concepts forming the path, or empty list if none found
        """
        visited = set()
        queue = deque([(concept_a, [concept_a])])

        while queue:
            current, path = queue.popleft()
            if current == concept_b:
                return path
            for neighbor in self.graph.get_connections(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return []

    def suggest_related(self, concept):
        """
        Suggests closely linked ideas based on the concept.
        :return: List of related nodes
        """
        return list(self.graph.get_connections(concept))

# Example usage
if __name__ == "__main__":
    from GraphBrainCore import GraphBrainCore

    brain = GraphBrainCore()
    brain.ingest("Elon Musk founded SpaceX and Neuralink", ["Elon Musk", "SpaceX", "Neuralink"])

    reasoner = GraphReasoner(brain)

    print("Are Elon Musk and Neuralink connected?", reasoner.path_exists("Elon Musk", "Neuralink"))
    print("Path between Elon and Neuralink:", reasoner.explain_connection("Elon Musk", "Neuralink"))
    print("What else is related to Elon?", reasoner.suggest_related("Elon Musk"))
