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

from typing import TYPE_CHECKING, Any, Dict

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


def generate_with_zero_shot_cot(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    input_text: str,
    cot_prompt: str = "Let's think step by step.\n",
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    **generate_kwargs,
) -> Dict[str, Any]:
    r"""
    Generate an answer from the LLM using zero-shot chain of thought (CoT) prompting.
    
    The zero-shot CoT approach encourages the model to reason step-by-step without any
    task-specific examples by injecting a reasoning prompt like "Let me think step by step".
    
    Args:
        model: The pretrained language model to use for generation
        tokenizer: The tokenizer corresponding to the model
        input_text: The original input/question to answer
        cot_prompt: The chain of thought prompt to inject (default: "Let's think step by step.\n")
        max_new_tokens: Maximum number of tokens to generate (default: 1024)
        temperature: Sampling temperature for generation (default: 0.7)
        top_p: Nucleus sampling parameter (default: 0.9)
        **generate_kwargs: Additional keyword arguments to pass to model.generate()
    
    Returns:
        A dictionary containing:
            - 'input_text': The original input
            - 'cot_prompt': The chain of thought prompt used
            - 'full_prompt': The complete prompt sent to the model
            - 'answer': The generated answer text
            - 'full_output': The complete output including the CoT prompt
    """
    # Combine input with CoT prompt
    # TODO: uncomment if dataset doesn't have cot_prompt
    # full_prompt = input_text + "\n" + cot_prompt
    full_prompt = input_text
    
    # Tokenize the input
    inputs = tokenizer(full_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)
    
    # Set default generation parameters
    print("temperature", temperature)

    generation_config = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": True if temperature > 0 else False,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    # Override with any additional kwargs provided
    generation_config.update(generate_kwargs)
    
    # Generate the answer
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
    
    # Decode the output
    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Extract just the answer part (after the CoT prompt)
    # The full_output contains both the input and the generated text
    answer_start_idx = full_output.find(cot_prompt)
    if answer_start_idx != -1:
        answer = full_output[answer_start_idx + len(cot_prompt):].strip()
    else:
        # Fallback if CoT prompt is not found in output
        answer = full_output.replace(input_text, "").strip()
    
    return {
        "input_text": input_text,
        "cot_prompt": cot_prompt,
        "full_prompt": full_prompt,
        "answer": answer,
        "full_output": full_output,
    }
