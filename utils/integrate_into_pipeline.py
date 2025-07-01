"""
File: integrate_into_pipeline.py
Purpose: Integration logic for FridayAI to query stored knowledge during runtime
"""

from utils.retrieve_knowledge import search_uploaded_knowledge

def query_knowledge(user_input: str):
    """
    Main entry for querying uploaded knowledge base using search.

    Args:
        user_input (str): User's question or message.

    Returns:
        list of dict: A list of matched passages with scores and filenames.
    """
    results = search_uploaded_knowledge(user_input)
    return results
