import argparse
import json
import os
import pdb
import sys

import jsonlines
import torch
from answer_extraction import extract_math_answer
from eval_script import eval_math

MAX_INT = sys.maxsize
INVALID_ANS = "[invalid]"

invalid_outputs = []

def test_hendrycks_math(path):
    questions = []
    answers = []
    preds = []
    
    
    with open(path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            questions.append(item['prompt'])
            temp_ans = item['label']
            answers.append(temp_ans)
            preds.append(item['predict'])

    results = []
    for idx, (question, pred, answer) in enumerate(zip(questions, preds, answers)):
        res = extract_math_answer(question, pred)
        results.append(res == answer)


    acc = sum(results) / len(results)
    # print('valid_outputs===', invalid_outputs)
    print(f"Accuracy: {acc:.2%}")




if __name__ == "__main__":

    test_hendrycks_math(path = "/Users/armin/Downloads/updated_predictions_generated_predictions_math_llama32_ttl.jsonl")