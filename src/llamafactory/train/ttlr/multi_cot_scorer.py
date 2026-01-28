# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Any, Dict, List

import torch
import torch.nn.functional as F
from tqdm import tqdm

 

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


def generate_multiple_cot_answers(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    input_text: str,
    num_generations: int = 5,
    cot_prompt: str = "Let's think step by step.\n",
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    **generate_kwargs,
) -> List[Dict[str, Any]]:
    r"""
    Generate multiple answers from the LLM using zero-shot chain of thought prompting.
    
    Args:
        model: The pretrained language model to use for generation
        tokenizer: The tokenizer corresponding to the model
        input_text: The original input/question to answer
        num_generations: Number of different answers to generate (default: 5)
        cot_prompt: The chain of thought prompt to inject (default: "Let's think step by step.\n")
        max_new_tokens: Maximum number of tokens to generate (default: 1024)
        temperature: Sampling temperature for generation (default: 0.7)
        top_p: Nucleus sampling parameter (default: 0.9)
        **generate_kwargs: Additional keyword arguments to pass to generate_with_zero_shot_cot()
    
    Returns:
        A list of dictionaries, each containing the generation result with keys:
            - 'input_text': The original input
            - 'cot_prompt': The chain of thought prompt used
            - 'full_prompt': The complete prompt sent to the model
            - 'answer': The generated answer text
            - 'full_output': The complete output including the CoT prompt
            - 'generation_idx': Index of this generation (0 to num_generations-1)
    """
    # Build full prompt by appending the CoT prompt to the input text so the model
    # is encouraged to produce a chain-of-thought and answer. Keep the prompt
    # structure compatible with the previous helper.
    #full_prompt = input_text + "\n" + cot_prompt
    full_prompt = input_text

    # Tokenize
    inputs = tokenizer(full_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    # Configure generation to return multiple independent samples in one call.
    # For independence we use sampling (do_sample=True) with the provided
    # temperature/top_p and request `num_return_sequences` samples.
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": True if temperature > 0 else False,
        "num_return_sequences": num_generations,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    generation_config.update(generate_kwargs)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            **generation_config,
        )

    # HuggingFace's generate returns a tensor with shape
    # (num_return_sequences, seq_len) for a single input. Decode each.
    generations = []
    for i, seq in enumerate(output_ids):
        full_output = tokenizer.decode(seq, skip_special_tokens=True)

        # Extract just the answer part (after the CoT prompt)
        answer_start_idx = full_output.find(cot_prompt)
        if answer_start_idx != -1:
            answer = full_output[answer_start_idx + len(cot_prompt) :].strip()
        else:
            # Fallback if CoT prompt is not found in output
            answer = full_output.replace(full_prompt, "").strip()

        generations.append(
            {
                "input_text": input_text,
                "cot_prompt": cot_prompt,
                "full_prompt": full_prompt,
                "answer": answer,
                "full_output": full_output,
                "generation_idx": i,
            }
        )

    return generations


def score_generation_plausibility(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    input_text: str,
    generated_text: str,
    scoring_strategy: str = "logits",
) -> float:
    r"""
    Score the plausibility of a generated text using different scoring strategies.
    
    Args:
        model: The pretrained language model to use for scoring
        tokenizer: The tokenizer corresponding to the model
        input_text: The original input/question
        generated_text: The generated answer to score
        scoring_strategy: The scoring strategy to use. Options:
            - "logits": Uses the model's loss (perplexity). Lower values indicate more plausible generations.
            - "cot": Uses the model to assign a score from 0 to 10 via chain-of-thought reasoning.
    
    Returns:
        A plausibility score. 
        - For "logits": negative log likelihood (lower is better)
        - For "cot": a score from 0 to 10 (higher is better)
    """
    if scoring_strategy == "logits":
        # Compute loss only over the generated_text tokens (ignore the input_text)
        # Tokenize input and generated text separately then concatenate so we can
        # mask the labels for the input portion.
        inputs_input = tokenizer(input_text, return_tensors="pt")
        inputs_gen = tokenizer(generated_text, return_tensors="pt")

        input_ids_input = inputs_input["input_ids"]
        input_ids_gen = inputs_gen["input_ids"]

        # Concatenate along sequence dimension
        input_ids = torch.cat([input_ids_input, input_ids_gen], dim=1).to(model.device)

        # Build attention mask if present
        attention_mask_input = inputs_input.get("attention_mask", None)
        attention_mask_gen = inputs_gen.get("attention_mask", None)
        if attention_mask_input is not None and attention_mask_gen is not None:
            attention_mask = torch.cat([attention_mask_input, attention_mask_gen], dim=1).to(model.device)
        else:
            attention_mask = None

        # Create labels that ignore the input_text portion (set to -100)
        labels = input_ids.clone()
        # Number of tokens in the input portion (per batch, here batch_size=1)
        input_len = input_ids_input.size(1)
        if labels.size(1) <= input_len:
            # No generated tokens present, return a large/neutral loss
            return float("inf")
        labels[:, :input_len] = -100

        # Compute loss only over generated tokens
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        return loss.item()
    
    elif scoring_strategy == "cot":
        # Use chain-of-thought prompting to score the generation
        scoring_prompt = (
            f"Input: {input_text}\n"
            f"Answer: {generated_text}\n"
            f"Let's think step by step about the quality and plausibility of this answer.\n"
            f"Please assign a score from 0 to 10 to this answer, where 0 means completely implausible "
            f"and 10 means perfectly plausible and accurate.\n"
            f"Score: "
        )
        
        # Tokenize the scoring prompt
        inputs = tokenizer(scoring_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)
        
        # Generate the score
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode the output
        score_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract the score from the generated text
        # Look for a number in the output
        score_str = score_text.replace(scoring_prompt, "").strip()
        
        # Try to extract the numeric score
        try:
            # Try to find a number in the response
            import re
            numbers = re.findall(r'\d+', score_str)
            if numbers:
                score = float(numbers[0])
                # Clamp score to [0, 10] range
                score = max(0.0, min(10.0, score))
                return score
            else:
                # If no number found, return a neutral score
                return 5.0
        except (ValueError, IndexError):
            # If parsing fails, return a neutral score
            return 5.0
    
    elif scoring_strategy == "yesno_logprob":
        """
        Ask the model whether the answer is high-quality (yes/no) and
        use log P(yes) - log P(no) as the plausibility score.
        """

        scoring_prompt = (
            f"Question: {input_text}\n"
            f"Answer: {generated_text}\n\n"
            f"Is this a high-quality, correct, and plausible answer?\n"
            f"Answer yes or no.\n"
        )

        # Tokenize prompt
        inputs = tokenizer(scoring_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)

        # Get logits for the *next token only*
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            # Logits for the next token position
            next_token_logits = outputs.logits[:, -1, :]  # shape: [1, vocab_size]
            log_probs = torch.log_softmax(next_token_logits, dim=-1)

        # Get token IDs for " yes" and " no"
        yes_token_id = tokenizer.encode(" yes", add_special_tokens=False)
        no_token_id = tokenizer.encode(" no", add_special_tokens=False)

        # Safety check: ensure single-token encoding
        if len(yes_token_id) != 1 or len(no_token_id) != 1:
            raise ValueError(
                "'yes' or 'no' is not a single token for this tokenizer. "
                "Consider alternative labels like 'true'/'false'."
            )

        yes_token_id = yes_token_id[0]
        no_token_id = no_token_id[0]

        # Extract log probabilities
        log_p_yes = log_probs[0, yes_token_id].item()
        log_p_no = log_probs[0, no_token_id].item()

        # Log-odds score
        score = log_p_yes - log_p_no
        return score


    else:
        raise ValueError(
            f"Unknown scoring_strategy: {scoring_strategy}. "
            f"Supported strategies are: 'logits', 'cot', 'yesno_logprob'."
        )


def generate_and_score_multiple_cot(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    input_text: str,
    num_generations: int = 5,
    cot_prompt: str = "Let's think step by step.\n",
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    scoring_strategy: str = "logits",
    **generate_kwargs,
) -> Dict[str, Any]:
    r"""
    Generate multiple zero-shot CoT answers and score their plausibility.
    
    This is a convenience function that combines generate_multiple_cot_answers()
    and score_generation_plausibility() into a single call.
    
    Args:
        model: The pretrained language model to use for generation and scoring
        tokenizer: The tokenizer corresponding to the model
        input_text: The original input/question to answer
        num_generations: Number of different answers to generate (default: 5)
        cot_prompt: The chain of thought prompt to inject (default: "Let's think step by step.\n")
        max_new_tokens: Maximum number of tokens to generate (default: 1024)
        temperature: Sampling temperature for generation (default: 0.7)
        top_p: Nucleus sampling parameter (default: 0.9)
        scoring_strategy: The scoring strategy to use. Options:
            - "logits": Uses the model's loss (perplexity). Lower values indicate more plausible generations.
            - "cot": Uses the model to assign a score from 0 to 10 via chain-of-thought reasoning.
        **generate_kwargs: Additional keyword arguments to pass to generation functions
    
    Returns:
        A dictionary containing:
            - 'input_text': The original input
            - 'generations': List of generated answers with keys:
                - 'answer': The generated answer text
                - 'full_output': The complete output including the CoT prompt
                - 'generation_idx': Index of this generation
                - 'plausibility_score': Plausibility score for this generation
            - 'best_generation_idx': Index of the generation with highest plausibility
            - 'best_answer': The most plausible generated answer
            - 'best_score': The plausibility score of the best answer
            - 'scores': List of all plausibility scores
            - 'scoring_strategy': The scoring strategy used
    """
    # Generate multiple answers
    generations = generate_multiple_cot_answers(
        model=model,
        tokenizer=tokenizer,
        input_text=input_text,
        num_generations=num_generations,
        cot_prompt=cot_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        **generate_kwargs,
    )
    
    # Score each generation
    scores = []
    for gen in generations:
        score = score_generation_plausibility(
            model=model,
            tokenizer=tokenizer,
            input_text=input_text,
            generated_text=gen["answer"],
            scoring_strategy=scoring_strategy,
        )
        gen["plausibility_score"] = score
        scores.append(score)
    
    # Find the best generation
    if scoring_strategy == "logits":
        # For logits, lower score is better
        best_idx = scores.index(min(scores))
    else:  # "cot" or "yesno_logprob"
        # For CoT, higher score is better
        best_idx = scores.index(max(scores))
    
    return {
        "input_text": input_text,
        "generations": generations,
        "best_generation_idx": best_idx,
        "best_answer": generations[best_idx]["answer"],
        "best_score": scores[best_idx],
        "scores": scores,
        "scoring_strategy": scoring_strategy,
    }


def create_cot_augmented_dataset(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    dataset,
    num_generations: int = 5,
    scoring_strategy: str = "logits",
    cot_prompt: str = "Let's think step by step.\n",
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    start_index: int = None,
    end_index: int = None,
) -> list:
    r"""
    Create a dataset augmented with best zero-shot CoT responses.
    
    For each sample in the input dataset, this function:
    1. Extracts the instruction/input from the sample
    2. Generates multiple CoT responses using the model
    3. Scores each response using the specified strategy
    4. Selects the best response
    5. Creates a new sample with the best CoT response as the output
    
    Args:
        model: The pretrained language model to use for generation and scoring
        tokenizer: The tokenizer corresponding to the model
        dataset: The input dataset (expects "instruction" and optionally "input" columns)
        num_generations: Number of different answers to generate per sample (default: 5)
        scoring_strategy: Strategy for scoring ("logits" or "cot", default: "logits")
        cot_prompt: The chain of thought prompt to inject (default: "Let's think step by step.\n")
        max_new_tokens: Maximum number of tokens to generate (default: 1024)
        temperature: Sampling temperature for generation (default: 0.7)
        start_index: Starting index for processing the dataset (default: None, process from beginning)
        end_index: Ending index for processing the dataset (default: None, process to end)
    
    Returns:
        A list of augmented dataset samples with best CoT responses
    """
    augmented_data = []
    print("SELECTED scoring_strategy", scoring_strategy)
    
    # Determine the range of indices to process
    if start_index is not None and end_index is not None:
        dataset_subset = dataset[start_index:end_index]
        print(f"Processing dataset from index {start_index} to {end_index}")
    else:
        dataset_subset = dataset
        print("Processing entire dataset")
    
    for idx, sample in enumerate(tqdm(dataset_subset, desc="Processing dataset samples")):
        # Extract instruction and input
        instruction = sample.get("instruction", "")
        query = sample.get("input", "")
        label = sample.get("output", "")
        print("query", query)
        
        # Combine instruction and input
        if query:
            input_text = f"{instruction}\n{query}"
        else:
            input_text = instruction
                
        try:
            # Generate and score multiple CoT responses
            result = generate_and_score_multiple_cot(
                model=model,
                tokenizer=tokenizer,
                input_text=input_text,
                num_generations=num_generations,
                cot_prompt=cot_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                scoring_strategy=scoring_strategy,
            )
            
            best_answer = result["best_answer"]
            best_score = result["best_score"]
            generations = result["generations"]
            
            # Create augmented sample with best CoT response
            augmented_sample = {
                **sample,  # Keep all original fields
                "cot_output": best_answer,
                "cot_best_score": best_score,
                "cot_num_generations": num_generations,
                "cot_strategy": scoring_strategy,
                "generations": generations,
                "label": label
            }
            
            augmented_data.append(augmented_sample)

            print("augmented_sample", augmented_sample)
        except Exception as e:
            # Fallback: keep original sample if generation fails
            augmented_data.append(sample)
    return augmented_data