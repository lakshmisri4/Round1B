import os
import json
from datetime import datetime
from pathlib import Path
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

# NLTK setup
nltk.download('punkt')
punkt_params = PunktParameters()
punkt_tokenizer = PunktSentenceTokenizer(punkt_params)

INPUT_JSON = "input/challenge1b_input.json"
OUTPUT_JSON = "output/final_output.json"

def load_input_metadata(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def collect_processed_data(documents):
    extracted_sections = []
    subsection_analysis_all = []

    for doc in documents:
        doc_name = doc["filename"]
        parsed_path = f"output/{doc_name}_parsed.json"

        if not os.path.exists(parsed_path):
            print(f"⚠️ Skipping missing: {parsed_path}")
            continue

        with open(parsed_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for idx, section in enumerate(data.get("sections", [])):
            heading = section.get("heading") or section.get("title") or "Untitled Section"
            page_num = section.get("page", 1)
            content = section.get("content", "").strip()

            extracted_sections.append({
                "document": doc_name,
                "section_title": heading,
                "importance_rank": idx + 1,
                "page_number": page_num
            })

            if content:
                sentences = punkt_tokenizer.tokenize(content)
                for sent in sentences:
                    if len(sent) > 40:
                        subsection_analysis_all.append({
                            "document": doc_name,
                            "refined_text": sent.strip(),
                            "page_number": page_num
                        })

    # Deduplicate by sentence & keep 5–10 diverse records from different PDFs
    unique_analysis = []
    seen_docs = set()
    for item in subsection_analysis_all:
        doc = item["document"]
        if doc not in seen_docs:
            unique_analysis.append(item)
            seen_docs.add(doc)
        if len(unique_analysis) >= 10:
            break

    return extracted_sections, unique_analysis

def save_final_output(input_data, extracted_sections, subsection_analysis):
    metadata = {
        "input_documents": [doc["filename"] for doc in input_data["documents"]],
        "persona": input_data.get("persona", {}).get("role", "Unknown"),
        "job_to_be_done": input_data.get("job_to_be_done", {}).get("task", "Unknown"),
        "processing_timestamp": datetime.now().isoformat()
    }

    top_sections = sorted(extracted_sections, key=lambda x: x["importance_rank"])[:5]

    output = {
        "metadata": metadata,
        "extracted_sections": top_sections,
        "subsection_analysis": subsection_analysis
    }

    Path("output").mkdir(exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Final output saved to {OUTPUT_JSON}")

def main():
    print("🚀 Starting final aggregation...")
    input_data = load_input_metadata(INPUT_JSON)
    extracted_sections, subsection_analysis = collect_processed_data(input_data["documents"])
    save_final_output(input_data, extracted_sections, subsection_analysis)

if __name__ == "__main__":
    main()
