# ======================================
# query_vector_index.py – Smart Search Tool (Profile-Aware)
# ======================================
#
# This script allows FridayAI to semantically search her indexed memory
# using a dynamic vector index file, with optional profile auto-loading.
#
# CMD Usage:
# python query_vector_index.py --query "Signs of labor?"
# python query_vector_index.py --query "How to change a timing belt?" --vector_file core/knowledge_data/memory/vector_index_mechanic.pkl
#
# Dependencies: sentence-transformers, numpy, pickle, argparse, json

import pickle
import argparse
import numpy as np
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load embedding model
MODEL_NAME = 'all-MiniLM-L6-v2'
model = SentenceTransformer(MODEL_NAME)

# Profile manager file
PROFILE_FILE = Path("core/knowledge_data/profile_manager.json")


def load_index(index_path):
    """Load the vector index from pickle file."""
    with open(index_path, 'rb') as f:
        return pickle.load(f)


def load_vector_path_from_profile():
    """Load default vector path based on current profile."""
    if not PROFILE_FILE.exists():
        print("[ERROR] profile_manager.json not found.")
        exit(1)
    with open(PROFILE_FILE, 'r', encoding='utf-8') as f:
        profile_data = json.load(f)
    current = profile_data.get("current_domain")
    return profile_data.get("profiles", {}).get(current, {}).get("vector_file")


def search(query, index_data, top_k=3):
    """
    Embed the query and perform cosine similarity against document embeddings.
    Return top_k most relevant documents.
    """
    query_vec = model.encode([query])
    doc_embeddings = np.array([np.array(doc['embedding']) for doc in index_data])

    similarities = cosine_similarity(query_vec, doc_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []
    for i in top_indices:
        results.append({
            'filename': index_data[i].get('filename', 'N/A'),
            'score': float(similarities[i]),
            'content': index_data[i]['content'][:500]  # Show preview only
        })
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Search vector index with a query")
    parser.add_argument('--query', type=str, required=True, help='Your question or topic')
    parser.add_argument('--top_k', type=int, default=3, help='Number of top results to return')
    parser.add_argument('--vector_file', type=str, help='Path to vector index file (overrides profile)')
    args = parser.parse_args()

    # Use passed path or profile default
    index_path = Path(args.vector_file) if args.vector_file else Path(load_vector_path_from_profile())
    if not index_path.exists():
        print(f"[ERROR] Index file not found at {index_path}")
        exit(1)

    index_data = load_index(index_path)

    # Search and display
    matches = search(args.query, index_data, args.top_k)

    print("\n[RESULTS]\n-------------------------------")
    for idx, result in enumerate(matches, 1):
        print(f"{idx}. FILE: {result['filename']} (score: {result['score']:.4f})")
        print(f"   {result['content'].replace('\n', ' ')[:500]}...\n")
