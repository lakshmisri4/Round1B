from keybert import KeyBERT

class KeyphraseExtractor:
    def __init__(self, model_path="models/all-MiniLM-L6-v2"):
        # Load local model (no internet)
        self.kw_model = KeyBERT(model=model_path)

    def extract_keywords(self, section_text, top_n=5):
        keywords = self.kw_model.extract_keywords(
            section_text,
            keyphrase_ngram_range=(1, 3),
            stop_words='english',
            top_n=top_n
        )
        return [phrase for phrase, score in keywords]
