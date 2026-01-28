import json


def main():
    # load data/AdaptEval/gsm8k_random_5k_raw.json
    original_path = "data/AdaptEval/gsm8k_random_5k_raw.json"
    with open(original_path, 'r') as f:
        original_data = json.load(f)
    

    # load each augmented_dataset_{start}_{end}.json
    augmented_paths = [
        "augmented_dataset_0_1000.json",
        "augmented_dataset_1000_2000.json",
        "augmented_dataset_2000_3000.json",
        "augmented_dataset_3000_4000.json",
        "augmented_dataset_4000_5000.json",
    ]

    # for each entry in the augmented dataset, add a field "label" which is the "output" field from the original dataset

    for augmented_path in augmented_paths:
        with open(augmented_path, 'r') as f:
            augmented_data = json.load(f)
        
        start_index = int(augmented_path.split('_')[-2])
        for i, entry in enumerate(augmented_data):
            original_entry = original_data[start_index + i]
            entry['label'] = original_entry['output']
        
        # save back to the same file
        with open(augmented_path, 'w') as f:
            json.dump(augmented_data, f, indent=2)

if __name__ == "__main__":
    main()