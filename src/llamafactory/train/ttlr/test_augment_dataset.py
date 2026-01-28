import json
import yaml
from typing import TYPE_CHECKING

from ...data import get_dataset, get_template_and_fix_tokenizer
from ...model import load_model, load_tokenizer
from .multi_cot_scorer import create_cot_augmented_dataset
from ...hparams import get_train_args  # Use get_train_args for full parsing

with open("examples/train_lora/offline_ttlr.yaml", 'r') as f:
    config = yaml.safe_load(f)
model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(config)

# Load dataset directly from JSON file
dataset_path = 'data/AdaptEval/gsm8k_random_5k_raw.json'  # Default path if not specified
print(dataset_path)
print((type(dataset_path)))
with open(dataset_path, 'r') as f:
    train_dataset = json.load(f)
print("49 train_dataset[0]", train_dataset[0])