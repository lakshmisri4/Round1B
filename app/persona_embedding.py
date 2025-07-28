from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json

def load_text(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def embed_text(text, model_path):
    model = SentenceTransformer(model_path)
    embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return embedding.tolist()

def save_embeddings(embedding_dict, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(embedding_dict, f, indent=2)
    print(f"Embeddings saved to: {output_file}")

def generate_persona_jd_embeddings(persona_path, jd_path, model_path, output_file):
    print("Loading Persona and JD...")
    persona_text = load_text(persona_path)
    jd_text = load_text(jd_path)

    print("Embedding Persona...")
    persona_vector = embed_text(persona_text, model_path)

    print("Embedding JD...")
    jd_vector = embed_text(jd_text, model_path)

    embedding_output = {
        "persona": {
            "text": persona_text,
            "embedding": persona_vector
        },
        "job_description": {
            "text": jd_text,
            "embedding": jd_vector
        }
    }

    save_embeddings(embedding_output, output_file)
