#!/usr/bin/env python3
"""
Script to load dataset and model, create augmented dataset using CoT, and save it.
"""

import argparse
import json
import yaml  # Added for loading YAML config
import torch  # Added for model generation test
from typing import TYPE_CHECKING

from ...data import get_dataset, get_template_and_fix_tokenizer
from ...model import load_model, load_tokenizer
from .multi_cot_scorer import create_cot_augmented_dataset
from ...hparams import get_train_args  # Use get_train_args for full parsing

if TYPE_CHECKING:
    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


def main():
    parser = argparse.ArgumentParser(description="Create and save CoT augmented dataset.")
    parser.add_argument("action", type=str, help="Action to perform (e.g., 'train')")
    parser.add_argument("config_file", type=str, help="Path to YAML config file")
    parser.add_argument("--start_index", type=int, help="Beginning index for dataset creation")
    parser.add_argument("--end_index", type=int, help="Ending index for dataset creation")

    args = parser.parse_args()

    # Load YAML config and parse args using get_train_args
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(config)
    print("model_args", model_args)
    print("data_args", data_args)

    # Load dataset directly from JSON file
    #dataset_path = getattr(data_args, 'dataset', 'data/AdaptEval/gsm8k_random_5k_raw.json')  # Default path if not specified
    dataset_path = 'data/AdaptEval/MetaMathQA_random_5k_qwen.json'
    # dataset_path = 'data/AdaptEval/math_5000_list.json'
    
    with open(dataset_path, 'r') as f:
        train_dataset = json.load(f)
    print("49 train_dataset[0]", train_dataset[0])
    # Load model (inference only)

    # Load tokenizer
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    model = load_model(tokenizer, model_args, finetuning_args, False)

    # # Test model generation to verify loading
    # test_prompt = "What is 2+2?"
    # inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    # with torch.no_grad():
    #     outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    # generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print(f"Test generation: {generated_text}")

    # Create augmented dataset (pull params from generating_args or use defaults)
    augmented_data = create_cot_augmented_dataset(
        model=model,
        tokenizer=tokenizer,
        dataset=train_dataset,
        num_generations=getattr(generating_args, 'num_generations', 1),
        scoring_strategy=getattr(generating_args, 'scoring_strategy', 'yesno_logprob'),
        cot_prompt=getattr(generating_args, 'cot_prompt', "Let's think step by step.\n"),
        max_new_tokens=getattr(generating_args, 'max_new_tokens', 1024),
        temperature=0.5,
        start_index=getattr(args, 'start_index', None),
        end_index=getattr(args, 'end_index', None),
    )

    # Save to JSON (assume output_file is in data_args or generating_args; add to YAML if needed)
    base_output_file = getattr(data_args, 'output_file', 'augmented_dataset_metamath_qwen.json')
    if hasattr(args, 'start_index') and args.start_index is not None and hasattr(args, 'end_index') and args.end_index is not None:
        # Insert start and end indices into the filename
        name_parts = base_output_file.rsplit('.', 1)
        output_file = f"{name_parts[0]}_{args.start_index}_{args.end_index}.{name_parts[1]}"
    else:
        output_file = base_output_file
    with open(output_file, 'w') as f:
        json.dump(augmented_data, f, indent=2)

    print(f"Augmented dataset saved to {output_file}")


if __name__ == "__main__":
    main()