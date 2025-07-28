# Persona-Driven Document Intelligence

## Problem Statement
- Extract the most relevant sections and refined sub-sections from a set of PDF documents.
- Output must align with a given persona and their job-to-be-done, formatted as a structured JSON.

## Approach
- A lightweight hybrid pipeline using semantic embeddings and keyword extraction.
- Each document is parsed, sectioned, embedded, and scored for relevance based on the persona+task query.
- Final output contains top 5 ranked sections and 5–10 refined sentence-level insights.

## Models and Libraries Used
- SentenceTransformers (MiniLM-L6-v2): To compute semantic similarity between document sections and the task.
- KeyBERT: For unsupervised keyword extraction from each section.
- scikit-learn (cosine_similarity): To rank sections by relevance to the persona’s task.
- PyMuPDF (fitz): To extract text with page layout.
- NLTK: For sentence tokenization.
- torch, numpy, json: For core computation and output formatting.

Each section is ranked using semantic similarity to the combined persona-task embedding vector. Top subsections are selected by filtering sentences with high keyword density and contextual alignment.
