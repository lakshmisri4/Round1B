from app.pdf_parser import extract_text_with_positions
from app.section_splitter import split_sections
from app.persona_embedding import generate_persona_jd_embeddings
from app.keyphrase_extractor import extract_keyphrases
from app.section_embedding import embed_sections, save_embeddings_to_file
import os
import json

def save_final_json(metadata, sections, keyphrases, output_file):
    output = {
        "metadata": metadata,
        "extracted_sections": [],
        "subsection_analysis": []
    }

    for i, section in enumerate(sections):
        output["extracted_sections"].append({
            "document": metadata["input_documents"][0],
            "section_title": section["title"],
            "importance_rank": i + 1,
            "page_number": section["page"]
        })

        output["subsection_analysis"].append({
            "document": metadata["input_documents"][0],
            "refined_text": section["content"],
            "page_number": section["page"],
            "keyphrases": keyphrases[i]
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"✅ Final output saved to: {output_file}")


def main():
    input_file = "input/sample_resume.pdf"
    output_file = "output/parsed_resume.json"
    model_path = "models/all-MiniLM-L6-v2"

    # Check input PDF file
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return

    print("🔍 Extracting raw text with positions...")
    parsed_data = extract_text_with_positions(input_file)

    metadata = {
        "input_documents": [os.path.basename(input_file)],
        "persona": "User Persona",
        "job_to_be_done": "Analyze and extract relevant sections"
    }

    print("📖 Splitting into logical sections using ML...")
    sections = split_sections(parsed_data, model_path)

    print("🧠 Extracting keyphrases for each section...")
    keyphrases = extract_keyphrases(sections, model_path)

    # NEW: Persona + JD Embedding Integration
    persona_file = "input/persona.txt"
    jd_file = "input/job_description.txt"
    embedding_output_file = "output/persona_jd_embeddings.json"

    if os.path.exists(persona_file) and os.path.exists(jd_file):
        print("🔗 Generating Persona + JD Embeddings...")
        generate_persona_jd_embeddings(persona_file, jd_file, model_path, embedding_output_file)
    else:
        print("⚠️ Persona or JD file not found, skipping embedding generation.")

    # ✅ NEW SECTION EMBEDDING
    print("🔎 Embedding sections with keyphrases...")
    section_embeddings = embed_sections(sections, keyphrases, model_path)
    save_embeddings_to_file(section_embeddings, "output/section_embeddings.json")

    print("📦 Saving final JSON output...")
    save_final_json(metadata, sections, keyphrases, output_file)

if __name__ == "__main__":
    main()
