# =====================================
# File: search_uploaded_knowledge.py
# Purpose: Query relevant information from uploaded knowledge base
# =====================================

import os
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

DATA_DIR = "core/knowledge_data"
VEC_PATH = os.path.join(DATA_DIR, "tfidf_vectorizer.pkl")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.pkl")

def load_vectorizer_and_corpus():
    with open(VEC_PATH, "rb") as vf:
        vectorizer = pickle.load(vf)
    with open(CORPUS_PATH, "rb") as cf:
        corpus = pickle.load(cf)
    return vectorizer, corpus

def query_knowledge(query: str, top_k: int = 5) -> List[Dict[str, str]]:
    vectorizer, corpus = load_vectorizer_and_corpus()
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, corpus["vectors"]).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            "source": corpus["sources"][idx],
            "text": corpus["texts"][idx],
            "score": float(similarities[idx])
        })
    return results

if __name__ == "__main__":
    while True:
        user_input = input("Ask something (or 'exit'): ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break
        results = query_knowledge(user_input)
        print("\nTop Results:")
        for r in results:
            print(f"- [{r['source']}] {r['text']} (score: {r['score']:.2f})")
