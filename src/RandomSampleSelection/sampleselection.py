import json
import random
import os

def sample_json_dataset(
    input_json_path,
    output_json_path,
    b,
    myseed
):
    # Set random seed for reproducibility
    random.seed(myseed)

    # Load original JSON file
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Sanity check
    if b > len(data):
        raise ValueError(f"b={b} is larger than dataset size={len(data)}")

    # Randomly sample b items
    sampled_data = random.sample(data, b)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    # Save sampled dataset with the same JSON format
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(sampled_data, f, indent=2, ensure_ascii=False)

    print(f"Saved {b} samples to {output_json_path}")


# =========================
# Usage
# =========================
if __name__ == "__main__":
    b_list = [25, 50, 100, 500, 1000, 5000]
    myseed = 2
    output_dir = "../../data/AdaptEval/SampleSelectionEx"

    input_json = "../../data/AdaptEval/SampleSelectionEx/fullset/MetaMathQA_augmented_noTemp_0_5000.json"
    for b in b_list:
        output_json = f"{output_dir}/RandomSelectionSeed{myseed}b{b}MetaMathQA.json"
        sample_json_dataset(
            input_json_path=input_json,
            output_json_path=output_json,
            b=b,
            myseed=myseed
        )

    input_json = "../../data/AdaptEval/SampleSelectionEx/fullset/logiqa_augmented_dataset_noTemp_5000.json"
    for b in b_list:
        output_json = f"{output_dir}/RandomSelectionSeed{myseed}b{b}LogiQA.json"
        sample_json_dataset(
            input_json_path=input_json,
            output_json_path=output_json,
            b=b,
            myseed=myseed
        )


