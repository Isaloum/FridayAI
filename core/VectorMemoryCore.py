# ================================================
# File: VectorMemoryCore.py
# Purpose: Semantic memory storage and retrieval using embeddings
# ================================================

import os
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util
import torch


class VectorMemoryCore:
    """
    A semantic memory system that stores embeddings of textual data
    and retrieves the most relevant entries based on cosine similarity.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Load the embedding model (SentenceTransformer)
        self.model = SentenceTransformer(model_name)
        self.entries: List[Dict] = []   # List to store text entries and metadata
        self.embeddings = None          # Torch tensor of all stored embeddings

    def ingest(self, text: str, metadata: Dict = None):
        """
        Embed and store a new text entry into the vector memory.

        :param text: Input string to store
        :param metadata: Optional metadata dict (tag, domain, etc.)
        """
        if not text or not isinstance(text, str):
            return

        # Encode the text into a vector (embedding)
        embedding = self.model.encode(text, convert_to_tensor=True)

        # Save entry and metadata
        self.entries.append({
            "text": text,
            "embedding": embedding,
            "metadata": metadata or {}
        })

        # Refresh the list of all embeddings
        self._update_embeddings()

    def query(self, prompt: str, top_k: int = 3, domain: str = None, mood: str = None) -> List[Dict]:
        """
        Semantic search with optional domain and mood filtering.
        """
      # print(f"\n[VectorMemory] Searching for: '{prompt}' (top_k={top_k}, domain={domain}, mood={mood})")

        if not self.entries:
           #print("[VectorMemory] No entries available.")
            return []

        filtered = self.entries
        if domain:
            filtered = [e for e in filtered if e["metadata"].get("domain") == domain]
        if mood:
            filtered = [e for e in filtered if e["metadata"].get("emotion") == mood]

        if not filtered:
           #print("[VectorMemory] No matching entries after filtering.")
            return []

        query_embedding = self.model.encode(prompt, convert_to_tensor=True)
        embeddings = torch.stack([e["embedding"] for e in filtered])
        cos_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
        top_results = np.argsort(-cos_scores.cpu().numpy())[:top_k]

        results = [{
            "text": filtered[i]["text"],
            "score": float(cos_scores[i]),
            "metadata": filtered[i]["metadata"]
        } for i in top_results]

       #print(f"[VectorMemory] Results: {results}")
        return results


    def _update_embeddings(self):
        """
        Refresh the internal tensor stack of all stored embeddings.
        """
        self.embeddings = torch.stack([e["embedding"] for e in self.entries])
