import json
import re

INPUT_FILE = "/Users/armin/Downloads/updated_predictions_generated_predictions_math_ttlr_qwen.jsonl"
OUTPUT_FILE = INPUT_FILE.replace(".jsonl", "_pruned.jsonl")
CHANGED_FILE = OUTPUT_FILE.replace(".jsonl", "_changed.jsonl")

BOXED_REGEX = re.compile(r"\\boxed\{[^}]+\}")
ANSWER_IS_REGEX = re.compile(r"answer is:\s*", re.IGNORECASE)

GARBAGE_STARTERS = ["Humans", "humans"]


def prune_prediction(text):
    # ---------- Rule 1: keep up to LAST \boxed ----------
    boxed_matches = list(BOXED_REGEX.finditer(text))
    if boxed_matches:
        last_boxed = boxed_matches[-1]
        return text[: last_boxed.end()].strip()

    # ---------- Rule 2: extract after "answer is:" ----------
    answer_match = ANSWER_IS_REGEX.search(text)
    if answer_match:
        remainder = text[answer_match.end():]

        # Cut at newline if present
        newline_idx = remainder.find("\n")
        if newline_idx != -1:
            remainder = remainder[:newline_idx]

        # Cut at garbage starters if present
        for g in GARBAGE_STARTERS:
            idx = remainder.find(g)
            if idx != -1:
                remainder = remainder[:idx]

        return remainder.strip()

    # ---------- Rule 3: hard prune on garbage starters ----------
    for g in GARBAGE_STARTERS:
        idx = text.find(g)
        if idx != -1:
            return text[:idx].strip()

    # ---------- Rule 4: fallback ----------
    return text.strip()


with open(INPUT_FILE, "r") as fin, \
     open(OUTPUT_FILE, "w") as fout, \
     open(CHANGED_FILE, "w") as fchanged:

    for line in fin:
        obj = json.loads(line)
        original = obj["predict"]
        pruned = prune_prediction(original)

        # Always write pruned output
        obj["predict"] = pruned
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

        # Log only changed predictions
        if pruned != original:
            change_record = {
                "prompt": obj.get("prompt"),
                "original_predict": original,
                "pruned_predict": pruned
            }
            fchanged.write(json.dumps(change_record, ensure_ascii=False) + "\n")

print(f"Pruned predictions written to: {OUTPUT_FILE}")
print(f"Changed predictions logged to: {CHANGED_FILE}")

