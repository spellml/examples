import argparse
import os
import shutil

from transformers.modeling_auto import AutoModelForQuestionAnswering
from transformers.tokenization_auto import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model")
args = parser.parse_args()

model = AutoModelForQuestionAnswering.from_pretrained(args.model)
tokenizer = AutoTokenizer.from_pretrained(args.model)

os.makedirs("model")
os.makedirs("tokenizer")

model.save_pretrained("model")
tokenizer.save_pretrained("tokenizer")

shutil.copyfile("model/config.json", "tokenizer/config.json")
