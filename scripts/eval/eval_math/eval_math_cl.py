#!/usr/bin/env python3
"""
Script to evaluate LLM predictions on the MATH dataset.
Extracts predicted answers from model responses and compares them against ground truth labels.
"""

import json
import re
import sys
from typing import List, Tuple, Optional


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract answer from \boxed{} LaTeX command.
    Handles nested braces.
    """
    # Find all \boxed{...} patterns
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    
    if matches:
        # Return the last boxed answer (usually the final answer)
        return matches[-1].strip()
    
    return None


def normalize_answer(answer: str) -> str:
    """
    Normalize an answer for comparison.
    - Remove whitespace
    - Convert to lowercase
    - Remove common LaTeX commands
    - Standardize notation
    """
    if not answer:
        return ""
    
    # Remove whitespace
    normalized = answer.strip()
    
    # Remove dollar signs
    normalized = normalized.replace('$', '')
    
    # Remove \text{} commands
    normalized = re.sub(r'\\text\{([^}]*)\}', r'\1', normalized)
    
    # Remove spaces
    normalized = normalized.replace(' ', '')
    
    # Remove backslashes before common symbols
    normalized = normalized.replace('\\', '')
    
    # Convert to lowercase for comparison
    normalized = normalized.lower()
    
    return normalized


def extract_final_number(text: str) -> Optional[str]:
    """
    Extract the final numerical answer from text.
    Looks for common patterns like "the answer is X", "= X", etc.
    """
    # Look for "the answer is X" or "answer: X" patterns
    patterns = [
        r'(?:the\s+)?(?:final\s+)?answer\s+is\s+[:\s]*([^\s.,;]+)',
        r'answer:\s*([^\s.,;]+)',
        r'therefore[,\s]+(?:the\s+)?(?:answer\s+is\s+)?([^\s.,;]+)',
        r'=\s*([^\s.,;]+)\s*$',  # Ends with = X
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            return matches[-1].strip()
    
    return None


def extract_answer(predict_text: str) -> Optional[str]:
    """
    Extract the predicted answer from model output.
    Tries multiple extraction strategies in order:
    1. \boxed{} command
    2. Common answer patterns
    3. Last number in text
    """
    # First, try to extract from \boxed{}
    boxed_answer = extract_boxed_answer(predict_text)
    if boxed_answer:
        return boxed_answer
    
    # Try to find explicit answer statements
    final_answer = extract_final_number(predict_text)
    if final_answer:
        return final_answer
    
    # As a fallback, try to extract the last number or expression
    # Look for numbers, fractions, or simple expressions at the end
    last_line = predict_text.strip().split('\n')[-1]
    
    # Look for number patterns (including fractions, decimals, negative numbers)
    number_patterns = [
        r'(-?\d+\.?\d*)',  # Decimal or integer
        r'(-?\d+/\d+)',    # Fraction
        r'(-?\d+)',        # Integer
    ]
    
    for pattern in number_patterns:
        matches = re.findall(pattern, last_line)
        if matches:
            return matches[-1].strip()
    
    return None


def compare_answers(predicted: Optional[str], labels: List[str]) -> bool:
    """
    Compare predicted answer against ground truth labels.
    Returns True if predicted matches any of the labels.
    """
    if predicted is None:
        return False
    
    normalized_pred = normalize_answer(predicted)
    
    for label in labels:
        normalized_label = normalize_answer(label)
        
        # Direct match
        if normalized_pred == normalized_label:
            return True
        
        # Check if they're equivalent numbers
        try:
            # Try to evaluate as numbers
            pred_num = eval(normalized_pred.replace('^', '**'))
            label_num = eval(normalized_label.replace('^', '**'))
            
            # Check if they're close (for floating point)
            if abs(pred_num - label_num) < 1e-6:
                return True
        except:
            pass
    
    return False


def evaluate_predictions(filepath: str, verbose: bool = False) -> Tuple[int, int, float]:
    """
    Evaluate predictions from a JSONL file.
    
    Args:
        filepath: Path to the JSONL file
        verbose: If True, print details for each example
    
    Returns:
        Tuple of (correct_count, total_count, accuracy)
    """
    correct = 0
    total = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                
                label = data.get('label', [])
                predict = data.get('predict', '')
                
                if not isinstance(label, list):
                    label = [label]
                
                # Extract predicted answer
                predicted_answer = extract_answer(predict)
                
                # Compare with labels
                is_correct = compare_answers(predicted_answer, label)
                
                if is_correct:
                    correct += 1
                
                total += 1
                
                if verbose:
                    status = "✓" if is_correct else "✗"
                    print(f"{status} Example {line_num}:")
                    print(f"  Ground truth: {label}")
                    print(f"  Predicted: {predicted_answer}")
                    print(f"  Normalized pred: {normalize_answer(predicted_answer) if predicted_answer else 'None'}")
                    print(f"  Normalized labels: {[normalize_answer(l) for l in label]}")
                    print()
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}", file=sys.stderr)
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}", file=sys.stderr)
                continue
    
    accuracy = (correct / total * 100) if total > 0 else 0.0
    
    return correct, total, accuracy


def eval_accuracy(paths):

    filepath = paths[0]
    verbose = True
    
    print(f"Evaluating predictions from: {filepath}")
    print("-" * 60)
    
    correct, total, accuracy = evaluate_predictions(filepath, verbose=verbose)
    
    print("-" * 60)
    print(f"Results:")
    print(f"  Correct: {correct}/{total}")
    print(f"  Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    paths = [
        # "generated_predictions_metamath_jinfer.jsonl",
        # "colota_updated_predictions.jsonl",
        "/Users/armin/Downloads/infeonlyllama31_math_5k_generated_predictions.jsonl"
    ]
    eval_accuracy(paths)