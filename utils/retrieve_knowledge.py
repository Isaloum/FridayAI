# =====================================
# File: retrieve_knowledge.py
# Purpose: Search relevant information from uploaded documents using vector similarity
# Author: FridayAI Developer
# =====================================

import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# === Load Preprocessed Data ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "../data/knowledge.pkl")
vectorizer_path = os.path.join(BASE_DIR, "../data/vectorizer.pkl")

with open(data_path, "rb") as f:
    knowledge = pickle.load(f)

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

# === Function to Search ===
def search_knowledge(query, top_n=5):
    """Search for top_n most relevant passages based on input query."""
    if not knowledge:
        print("Knowledge base is empty.")
        return []

    # Convert stored content into TF-IDF vector space
    documents = [item["text"] for item in knowledge]
    doc_vectors = vectorizer.transform(documents)

    # Transform the input query into the same vector space
    query_vec = vectorizer.transform([query])

    # Compute cosine similarity between query and all documents
    similarities = cosine_similarity(query_vec, doc_vectors).flatten()

    # Get top_n indices sorted by similarity
    top_indices = similarities.argsort()[::-1][:top_n]

    results = []
    for idx in top_indices:
        results.append({
            "score": round(similarities[idx], 4),
            "text": knowledge[idx]["text"],
            "source": knowledge[idx]["source"]
        })

    return results

# === Command-line Usage ===
if __name__ == "__main__":
    while True:
        query = input("\n🔍 Enter a question (or 'exit'): ").strip()
        if query.lower() in ["exit", "quit"]:
            break
        matches = search_knowledge(query)
        if not matches:
            print("No relevant information found.")
        else:
            print("\n📚 Top Results:")
            for i, match in enumerate(matches, 1):
                print(f"\n{i}. ({match['score']}) {match['text']}\n   — Source: {match['source']}")
