import fitz  
import os
import json

def extract_text_with_positions(pdf_path):
    doc = fitz.open(pdf_path)
    extracted_data = []

    for page_number in range(len(doc)):
        page = doc[page_number]
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if "lines" in block: 
                text = ""
                for line in block["lines"]:
                    for span in line["spans"]:
                        text += span["text"] + " "
                text = text.strip()

                if text:
                    extracted_data.append({
                        "text": text,
                        "page_number": page_number + 1,
                        "document": os.path.basename(pdf_path)
                    })

    return extracted_data


def save_as_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
