from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os

def embed_sections(sections, keyphrases, model_path):
    model = SentenceTransformer(model_path)

    embeddings = []
    for section, phrases in zip(sections, keyphrases):
        combined_text = section["content"] + " " + " ".join(phrases)
        vector = model.encode(combined_text)
        embeddings.append({
            "section_title": section["title"],
            "page_number": section["page"],
            "embedding": vector.tolist()  
        })

    return embeddings

def save_embeddings_to_file(embeddings, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(embeddings, f, indent=2)
    print(f"Section embeddings saved to: {output_path}")
