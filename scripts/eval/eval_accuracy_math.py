import json
import re

# --- MATH-500 SPECIFIC UTILS ---

def normalize_math_text(text: str) -> str:
    """Standardizes LaTeX strings for comparison."""
    if not text:
        return ""
    # Remove common LaTeX formatting that doesn't change value
    text = text.replace(r"\left", "").replace(r"\right", "")
    text = text.replace(r"\text", "").replace("{", "").replace("}", "")
    text = text.replace(r"\,", "").replace(r" ", "")
    
    # Standardize specific symbols
    text = text.replace("pi", "π").replace(r"\pi", "π")
    
    # Handle fractions (simple case: \frac{a}{b} -> a/b)
    text = re.sub(r"\\frac(\d)(\d)", r"\1/\2", text)
    
    return text.strip().lower()

def extract_math_answer(completion: str) -> str:
    """Extracts answer from \boxed{} or falls back to the last sequence."""
    # 1. Look for \boxed{...}
    boxed = re.findall(r"\\boxed\{(.*?)\}", completion)
    if boxed:
        return boxed[-1]
    
    # 2. Fallback: Last line or last numeric/math sequence
    lines = completion.strip().split('\n')
    last_line = lines[-1].replace('$', '').strip()
    # Try to find 'The answer is X' pattern
    match = re.search(r"is\s+([-+]?\d*\.?\d+.*)$", last_line, re.IGNORECASE)
    if match:
        return match.group(1)
    return last_line

def math_equal(prediction: str, label: str) -> bool:
    """Checks if predicted math string matches ground truth."""
    pred_norm = normalize_math_text(prediction)
    label_norm = normalize_math_text(label)
    
    if pred_norm == label_norm:
        return True
    
    # Basic numeric equivalence (e.g., 0.5 == 1/2)
    try:
        # Warning: eval is used here for brevity; in production, use a safe math parser
        if '/' in label_norm or '/' in pred_norm:
            return float(eval(pred_norm.replace('π', '3.14159'))) == \
                   float(eval(label_norm.replace('π', '3.14159')))
    except:
        pass
        
    return False

# --- MODIFIED CORE SCRIPT ---

def eval_accuracy(paths):
    ems = []
    for path in paths:
        all_result = []
        labels = []
        with open(path, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                data = json.loads(line)
                completion = data["predict"]
                label = data["label"]

                # Logic for MATH-500
                if "math" in path.lower():
                    result = extract_math_answer(completion)
                else:
                    # Keep your original logic for GSM8K
                    from eval_utils import extract_gsm8k_answer_number
                    result = str(extract_gsm8k_answer_number(completion))
                
                all_result.append(result)
                labels.append(label.strip())
            
            # Use math_equal instead of 'a == b' for flexible matching
            compare_res = [math_equal(a, b) for a, b in zip(all_result, labels)]
            em = sum(compare_res) / len(all_result)
            ems.append(em)
    
    for path, em in zip(paths, ems):
        print(f"Path: {path}")
        print(f"Accuracy: {em:.2%}")           

if __name__ == "__main__":
    paths = [
        "/Users/armin/Downloads/generated_predictions_ttl_math5k.jsonl",
    ]
    eval_accuracy(paths)