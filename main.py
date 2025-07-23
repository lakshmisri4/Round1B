from app.pdf_parser import extract_text_with_positions
from app.section_splitter import split_sections
from app.keyphrase_extractor import extract_keyphrases
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

    print("📦 Saving final JSON output...")
    save_final_json(metadata, sections, keyphrases, output_file)

if __name__ == "__main__":
    main()
