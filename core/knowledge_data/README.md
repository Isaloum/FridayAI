# Knowledge Data Folder

This folder contains uploaded and ingested knowledge sources to be used by FridayAI.

## Purpose
To provide a structured and expandable foundation of domain-specific knowledge (e.g., pregnancy, engineering, law) that can be searched and reasoned over by FridayAI.

## Structure
Each file added here should be in `.txt`, `.pdf`, or `.md` format and will be processed by ingestion tools like `upload_knowledge.py`.

## Ingestion Strategy
The ingestion tool extracts clean, semantically relevant content from these sources and embeds it using vector representations to enable intelligent searching and retrieval.

## Usage
Upload content here manually or via scripts. After upload, run:
```bash
python upload_knowledge.py
