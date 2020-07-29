from spell.serving import BasePredictor

from transformers import pipeline
from transformers.modeling_auto import AutoModelForQuestionAnswering
from transformers.tokenization_auto import AutoTokenizer


class Predictor(BasePredictor):
    def __init__(self):
        model = AutoModelForQuestionAnswering.from_pretrained("/model/model")
        tokenizer = AutoTokenizer.from_pretrained("/model/tokenizer")
        self.default_response = "Perhaps the answer is 42."
        self.predictor = pipeline(
            "question-answering", model=model, tokenizer=tokenizer
        )
        with open("/mounts/bert_context/paragraph.txt") as f:
            self.context = f.read()

    def predict(self, payload):
        question = payload["question"]
        context = payload.get("context", self.context)

        try:
            res = self.predictor({"question": question, "context": context})
        except Exception:
            res = self.default_response
        return res
