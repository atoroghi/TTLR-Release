import json
import os
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import re
def greedy_k_center(
    embeddings: List[torch.Tensor],
    b: int,
    robust: bool = False,
    epsilon: float = 0.05,
    device: Optional[torch.device] = None
) -> List[int]:
    """
    Greedy k-center / robust k-center selection.
    """
    assert 0 <= epsilon < 1
    if device is None:
        device = embeddings[0].device

    X = torch.stack(embeddings).to(device)  # (N, d)
    N = X.shape[0]
    assert b <= N

    # Initialize with farthest-from-mean (stable)
    mean = X.mean(dim=0)
    distances = torch.norm(X - mean, dim=1)
    first_idx = torch.argmax(distances).item()

    selected = [first_idx]
    distances = torch.norm(X - X[first_idx], dim=1)

    trim_k = int(epsilon * N)

    for _ in range(1, b):
        if robust and trim_k > 0:
            threshold, _ = torch.kthvalue(distances, k=N - trim_k)
            mask = distances <= threshold
            masked_distances = distances.clone()
            masked_distances[~mask] = -1.0
            next_idx = torch.argmax(masked_distances).item()
        else:
            next_idx = torch.argmax(distances).item()

        selected.append(next_idx)
        new_distances = torch.norm(X - X[next_idx], dim=1)
        distances = torch.minimum(distances, new_distances)

    return selected


def greedy_facility_location(
    embeddings: List[torch.Tensor],
    b: int,
    device: Optional[torch.device] = None,
) -> List[int]:
    """
    Greedy facility location selection.
    Assumes embeddings are L2-normalized.
    """

    if device is None:
        device = embeddings[0].device

    X = torch.stack(embeddings).to(device)  # (N, d)
    N = X.shape[0]
    assert b <= N

    # Similarity matrix (cosine similarity)
    sim = X @ X.T  # (N, N)

    # Current best similarity to selected set
    coverage = torch.zeros(N, device=device)

    selected = []
    remaining = torch.ones(N, dtype=torch.bool, device=device)

    for _ in range(b):
        # Marginal gain: sum(max(sim[:, j], coverage) - coverage)
        gains = torch.sum(torch.maximum(coverage.unsqueeze(1), sim) - coverage.unsqueeze(1), dim=0)

        gains[~remaining] = -float("inf")
        next_idx = torch.argmax(gains).item()

        selected.append(next_idx)
        remaining[next_idx] = False
        coverage = torch.maximum(coverage, sim[:, next_idx])

    return selected

import json
import torch
import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import re


def extract_question(text: str) -> str:
    """
    Extracts the question from different prompt formats.
    """
    pattern_instruction = r"### Instruction:\s*(.*?)\s*### Response:"
    match = re.search(pattern_instruction, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    pattern_cot = r"^(.*?)\n\nLet's think step by step\."
    match = re.search(pattern_cot, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    pattern_answer = r"^(.*?)\n\nAnswer:"
    match = re.search(pattern_answer, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    raise ValueError("Could not extract question from the provided text.")


def select_subset(
    input_json_path: str,
    output_json_path: str,
    b: int,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    selection_strategy: str = "kcenter",
    robust: bool = False,
    epsilon: float = 0.05,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Load dataset, embed input_text, select subset, and save it.
    """

    # ------------------------
    # Load dataset
    # ------------------------
    with open(input_json_path, "r") as f:
        dataset = json.load(f)

    texts = [extract_question(entry["instruction"]) for entry in dataset]
    N = len(texts)
    assert b <= N

    # ------------------------
    # Load embedding model (NOT TIMED)
    # ------------------------
    model = SentenceTransformer(embedding_model_name, device=device)

    # ------------------------
    # Embed texts (TIMED)
    # ------------------------
    if device == "cuda":
        torch.cuda.synchronize()
    embed_start = time.perf_counter()

    embeddings = []
    for i in tqdm(range(0, N, batch_size), desc="Embedding"):
        with torch.no_grad():
            batch_emb = model.encode(
                texts[i:i + batch_size],
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
        embeddings.extend(batch_emb)

    if device == "cuda":
        torch.cuda.synchronize()
    embed_time = time.perf_counter() - embed_start

    # ------------------------
    # Selection (TIMED)
    # ------------------------
    if device == "cuda":
        torch.cuda.synchronize()
    select_start = time.perf_counter()

    if selection_strategy == "kcenter":
        selected_indices = greedy_k_center(
            embeddings=embeddings,
            b=b,
            robust=robust,
            epsilon=epsilon,
            device=torch.device(device),
        )

    elif selection_strategy == "facility_location":
        selected_indices = greedy_facility_location(
            embeddings=embeddings,
            b=b,
            device=torch.device(device),
        )

    else:
        raise ValueError(f"Unknown selection strategy: {selection_strategy}")

    if device == "cuda":
        torch.cuda.synchronize()
    selection_time = time.perf_counter() - select_start

    selected_indices = sorted(selected_indices)

    # ------------------------
    # Extract + save
    # ------------------------
    subset = [dataset[i] for i in selected_indices]

    with open(output_json_path, "w") as f:
        json.dump(subset, f, indent=2)

    # ------------------------
    # Report timings
    # ------------------------
    print(f"Saved {len(subset)} samples to {output_json_path}")
    print(f"Embedding time: {embed_time:.2f} seconds")
    print(f"Selection time: {selection_time:.2f} seconds")



def select_subset_random(
    input_json_path: str,
    output_json_path: str,
    b: int,
    seed: Optional[int] = None,
):
    """
    Load dataset, randomly select b samples, and save subset.
    """

    # ------------------------
    # Load dataset
    # ------------------------
    with open(input_json_path, "r") as f:
        dataset = json.load(f)

    assert isinstance(dataset, list), "Dataset must be a list of JSON objects"
    N = len(dataset)
    assert b <= N, f"Selection budget b={b} exceeds dataset size N={N}"

    # ------------------------
    # Optional reproducibility
    # ------------------------
    if seed is not None:
        random.seed(seed)

    # ------------------------
    # Random selection
    # ------------------------
    selected_indices: List[int] = random.sample(range(N), b)
    selected_indices.sort()

    # ------------------------
    # Extract subset
    # ------------------------
    subset = [dataset[i] for i in selected_indices]

    # ------------------------
    # Save subset
    # ------------------------
    with open(output_json_path, "w") as f:
        json.dump(subset, f, indent=2)

    print(f"Saved {len(subset)} randomly selected samples to {output_json_path}")


if __name__ == "__main__":
    print("current dir", os.getcwd())

    select_subset(
    # input_json_path="gsm8k_augmented_dataset_0_5000.json",
    input_json_path="data/AdaptEval/augmented_dataset_math_0_5000_pruned.json",
    # output_json_path="selected100_facility_math_augmented_dataset_0_5000_Qwen3-Embedding-0.6B.json",
    output_json_path="timing-test.json",
    b=100,
    selection_strategy="facility_location",
    embedding_model_name="/scratch/atoroghi/Qwen3-Embedding-0.6B"
)