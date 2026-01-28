from .eval_utils import extract_gsm8k_answer_number, extract_math_answer, extract_logiqa_option
import json


def eval_accuracy(paths):
    ems = []
    for path in paths:
        all_result = []
        labels = []
        with open(path, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                data = json.loads(line)
                completion = data["predict"]
                if "logiqa" in path:
                    result = extract_logiqa_option(completion)  # logiqa dataset
                elif "gsm8k" in path:
                    result = extract_gsm8k_answer_number(completion)  # gsm8k dataset
                elif "metamath" in path:
                    result = extract_math_answer(completion, data["label"]) # meta_math dataset
                # result = extract_logiqa_option(completion)  # logiqa dataset
                # result = extract_gsm8k_answer_number(completion)  # gsm8k dataset
                # result = extract_math_answer(completion, data["label"]) # meta_math dataset
                all_result.append(result)
                labels.append(data["label"].strip('\n'))   # .replace(',', '')
            
            compare_res = [a == b for a, b in zip(all_result, labels)]
            em = sum(compare_res) / len(all_result)
            ems.append(em)
    
    for path, em in zip(paths, ems):
        print(path)
        print(f"Accuracy: {em}")           
    
if __name__ == "__main__":
    paths = [
        "saves/llama3-8b/offline_ttlr/gsm8k/Temp_1000/generated_predictions.jsonl",
    ]
    eval_accuracy(paths)