import os
from app.pdf_parse import extract_text_with_positions, save_as_json
from app.persona_embedding import generate_persona_jd_embeddings
from app.keyphrase_extractor import KeyphraseExtractor
from app.section_splitter import split_sections  # You must define and expose this
from app.output_utils import save_final_json  # If not present, create it
from app.output_utils import save_final_json


def main():
    input_folder = "input"
    output_folder = "output"
    model_path = "models/all-MiniLM-L6-v2"

    persona_file = os.path.join(input_folder, "persona.txt")
    jd_file = os.path.join(input_folder, "job_description.txt")
    embedding_output_file = os.path.join(output_folder, "persona_jd_embeddings.json")

    if os.path.exists(persona_file) and os.path.exists(jd_file):
        print("🔗 Generating Persona + JD Embeddings...")
        generate_persona_jd_embeddings(persona_file, jd_file, model_path, embedding_output_file)
    else:
        print("⚠️ Persona or JD file not found, skipping embedding generation.")

    for filename in os.listdir(input_folder):
        if filename.endswith(".pdf"):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, f"{filename}_parsed.json")

            print(f"\n📄 Processing: {filename}")
            parsed_data = extract_text_with_positions(input_file)

            metadata = {
                "input_documents": [os.path.basename(input_file)],
                "persona": "User Persona",
                "job_to_be_done": "Analyze and extract relevant sections"
            }

            print("📖 Splitting into sections...")
            sections = split_sections(parsed_data, model_path)

            print("🧠 Extracting keyphrases...")
            extractor = KeyphraseExtractor(model_path)
            keyphrases = [extractor.extract_keywords(section["content"]) for section in sections]


            print("🔎 Embedding sections...")
            from app.section_embedding import embed_sections, save_embeddings_to_file
            section_embeddings = embed_sections(sections, keyphrases, model_path)

            save_embeddings_to_file(section_embeddings, os.path.join(output_folder, f"{filename}_embeddings.json"))

            print("📦 Saving JSON output...")
            save_final_json(metadata, sections, keyphrases, output_file)


    print("\n✅ All PDFs processed.")