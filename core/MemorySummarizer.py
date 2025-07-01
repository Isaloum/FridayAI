# ==============================================
# File: Core.MemorySummarizer.py
# Purpose: Summarize top vector hits into one reflective insight
# ==============================================

from typing import List

class MemorySummarizer:
    @staticmethod
    def summarize_vector_hits(hits: List[dict]) -> str:
        if not hits:
            return "No prior context available."

        texts = [h["text"] for h in hits]
        if len(texts) == 1:
            return f"Earlier memory shows: {texts[0]}"
        
        # Crude but fast summarizer — can be upgraded to LLM later
        keywords = []
        for text in texts:
            words = [w.lower() for w in text.split() if len(w) > 4]
            keywords.extend(words)

        top_words = sorted(set(keywords), key=keywords.count, reverse=True)[:4]
        summary = " ".join(top_words)
        return f"Past patterns indicate relevance in: {summary}"
