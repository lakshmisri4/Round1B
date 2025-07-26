import fitz  # PyMuPDF
import os
import json
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import numpy as np

# Paths
PDF_PATH = "input/sample_resume.pdf"
MODEL_PATH = "models/all-MiniLM-L6-v2"
OUTPUT_PATH = "output/sections.json"

# Load Sentence Transformer Model
model = SentenceTransformer('all-MiniLM-L6-v2')
model.save('models/all-MiniLM-L6-v2') # Offline model

def extract_text_blocks(pdf_path):
    doc = fitz.open(pdf_path)
    text_blocks = []

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("blocks")  # List of (x0, y0, x1, y1, "text", block_no)
        for block in blocks:
            text = block[4].strip()
            if len(text) > 20:  # Filter out noise
                text_blocks.append({
                    "text": text,
                    "page_number": page_num
                })
    return text_blocks

def cluster_text_blocks(text_blocks, n_clusters=6):
    texts = [block["text"] for block in text_blocks]
    embeddings = model.encode(texts)

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    # Group by cluster
    sections = {}
    for label, block in zip(labels, text_blocks):
        if label not in sections:
            sections[label] = []
        sections[label].append(block)

    return sections

def save_sections_json(sections, output_path):
    output_data = []
    for rank, (label, blocks) in enumerate(sorted(sections.items())):
        combined_text = " ".join([b["text"] for b in blocks])
        pages = list(set(b["page_number"] for b in blocks))
        output_data.append({
            "section_id": label,
            "importance_rank": rank + 1,
            "pages": sorted(pages),
            "content": combined_text
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

# section_splitter.py
def split_sections(parsed_data, model_path, n_clusters=6):
    model = SentenceTransformer(model_path)
    texts = [item["text"] for item in parsed_data]
    embeddings = model.encode(texts)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    sections = []
    for i in range(n_clusters):
        section_texts = [texts[j] for j in range(len(labels)) if labels[j] == i]
        pages = [parsed_data[j]["page_number"] for j in range(len(labels)) if labels[j] == i]
        sections.append({
            "title": f"Section {i+1}",
            "content": " ".join(section_texts),
            "page": min(pages) if pages else 1
        })

    return sections


def main():
    os.makedirs("output", exist_ok=True)
    print("[+] Extracting text blocks...")
    blocks = extract_text_blocks(PDF_PATH)
    print(f"[✓] Found {len(blocks)} blocks")

    print("[+] Clustering into sections...")
    sections = cluster_text_blocks(blocks, n_clusters=6)

    print(f"[✓] Extracted {len(sections)} sections")
    save_sections_json(sections, OUTPUT_PATH)
    print(f"[✓] Output saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
