import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from typing import List, Dict


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
MODEL_DIR = "/path/to/llama-3-8b"
DATASET_PATH = "gsm8k.json"
OUTPUT_PATH = "gsm8k_selected.json"

QUESTION_KEY = "question"
TOP_B = 1000

UNCERTAINTY_METRIC = "entropy" | "nll" | "answer_entropy" | "bald"
MC_DROPOUT_PASSES = 8   # 5–10 is usually enough


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


# ---------------------------------------------------------
# Load model and tokenizer
# ---------------------------------------------------------
def load_llm(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=DTYPE,
        device_map="auto" if DEVICE == "cuda" else None
    )
    model.eval()
    return tokenizer, model


# ---------------------------------------------------------
# Load JSON dataset
# ---------------------------------------------------------
def load_dataset(path: str) -> List[Dict]:
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------
# Predictive entropy
# ---------------------------------------------------------
@torch.no_grad()
def compute_entropy(text: str, tokenizer, model) -> float:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True
    ).to(model.device)

    outputs = model(**inputs)
    logits = outputs.logits[:, :-1, :]  # next-token logits

    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)

    return entropy.mean().item()


# ---------------------------------------------------------
# Sequence NLL
# ---------------------------------------------------------
@torch.no_grad()
def compute_sequence_nll(text: str, tokenizer, model) -> float:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True
    ).to(model.device)

    input_ids = inputs["input_ids"]

    outputs = model(**inputs)
    logits = outputs.logits[:, :-1, :]
    targets = input_ids[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1,
        index=targets.unsqueeze(-1)
    ).squeeze(-1)

    nll = -token_log_probs.mean()

    return nll.item()

@torch.no_grad()
def compute_first_answer_entropy(
    question: str,
    tokenizer,
    model
) -> float:
    """
    Computes entropy of p(y_1 | question)
    """

    inputs = tokenizer(
        question,
        return_tensors="pt",
        truncation=True
    ).to(model.device)

    outputs = model(**inputs)

    # Last token logits → next-token distribution
    next_token_logits = outputs.logits[:, -1, :]  # [1, V]
    probs = F.softmax(next_token_logits, dim=-1)

    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)

    return entropy.item()

@torch.no_grad()
def compute_bald_first_answer_token(
    question: str,
    tokenizer,
    model,
    mc_passes: int
) -> float:
    """
    BALD = H(mean p) - mean H(p)
    for the first generated answer token
    """

    inputs = tokenizer(
        question,
        return_tensors="pt",
        truncation=True
    ).to(model.device)

    # Enable dropout
    model.train()

    probs_list = []

    for _ in range(mc_passes):
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]  # next-token logits
        probs = F.softmax(logits, dim=-1)
        probs_list.append(probs)

    probs_stack = torch.stack(probs_list, dim=0)  # [K, 1, V]

    # Mean predictive distribution
    mean_probs = probs_stack.mean(dim=0)

    # H[p(y | x)]
    entropy_mean = -(mean_probs * torch.log(mean_probs + 1e-12)).sum(dim=-1)

    # E[H[p(y | x, θ)]]
    entropy_each = -(probs_stack * torch.log(probs_stack + 1e-12)).sum(dim=-1)
    expected_entropy = entropy_each.mean(dim=0)

    bald = entropy_mean - expected_entropy

    # Return scalar
    return bald.item()



# ---------------------------------------------------------
# Unified scoring
# ---------------------------------------------------------
def score_dataset(
    dataset,
    tokenizer,
    model,
    question_key,
    metric
):
    scored = []

    for item in tqdm(dataset, desc=f"Scoring with {metric}"):
        question = item[question_key]

        if metric == "entropy":
            score = compute_entropy(question, tokenizer, model)

        elif metric == "nll":
            score = compute_sequence_nll(question, tokenizer, model)

        elif metric == "answer_entropy":
            score = compute_first_answer_entropy(
                question, tokenizer, model
            )

        elif metric == "bald":
            score = compute_bald_first_answer_token(
                question,
                tokenizer,
                model,
                MC_DROPOUT_PASSES
            )

        else:
            raise ValueError(f"Unknown metric: {metric}")

        scored.append((score, item))

    return scored




# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------
def main():
    tokenizer, model = load_llm(MODEL_DIR)
    dataset = load_dataset(DATASET_PATH)

    scored_samples = score_dataset(
        dataset,
        tokenizer,
        model,
        QUESTION_KEY,
        UNCERTAINTY_METRIC
    )

    # Higher = more uncertain / surprising
    scored_samples.sort(key=lambda x: x[0], reverse=True)

    selected = [item for _, item in scored_samples[:TOP_B]]

    with open(OUTPUT_PATH, "w") as f:
        json.dump(selected, f, indent=2)

    print(
        f"Saved {len(selected)} samples "
        f"(metric={UNCERTAINTY_METRIC}) → {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
