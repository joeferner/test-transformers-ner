from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

class NerModel:
    def __init__(self):
        self._tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER", local_files_only=True)
        self._model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER", local_files_only=True)

    def extract(self, text: str):
        nlp = pipeline("ner", model=self._model, tokenizer=self._tokenizer)
        return nlp(text)

model = NerModel()

def get_ner_model():
    return model
