import json

def save_final_json(metadata, sections, keyphrases, output_path):
    output = {
        "metadata": metadata,
        "sections": []
    }

    for section, phrases in zip(sections, keyphrases):
        output["sections"].append({
            "title": section["title"],
            "page_number": section["page"],
            "content": section["content"],
            "keyphrases": phrases
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
