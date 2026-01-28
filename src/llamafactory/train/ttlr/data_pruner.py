# import json
# import re
# from typing import List, Dict


# def prune_llm_outputs(json_path: str,
#                       stop_markers: List[str] = None,
#                       keep_final_answer: bool = True) -> List[Dict]:
#     """
#     Load a JSON file (list of dicts), prune over-generated 'output' fields,
#     and return the cleaned list.

#     Args:
#         json_path: Path to the JSON file.
#         stop_markers: List of strings that indicate the start of unwanted text.
#         keep_final_answer: If True, keep text up to and including the first
#                            'Final Answer:' segment.

#     Returns:
#         A list of dictionaries with pruned 'output' fields.
#     """

#     if stop_markers is None:
#         stop_markers = [
#             "### Instruction:",
#             "### Response:",
#         ]

#     with open(json_path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     for entry in data:
#         if "output" not in entry or not isinstance(entry["output"], str):
#             continue

#         text = entry["output"]
#         entry["original_output"] = text 

#         final_answer_index = re.search(r"final answer:", text, re.IGNORECASE)
#         if final_answer_index:
#             period_index = text.find(".", final_answer_index.end())
#             if period_index != -1:
#                 text = text[:period_index + 1]

#         cut_positions = [
#             text.find(marker) for marker in stop_markers if marker in text
#         ]

#         if cut_positions:
#             text = text[:min(cut_positions)]

#         entry["output"] = text.strip()

    
#     final_path = json_path.replace(".json", "_pruned.json")
#     with open(final_path, "w", encoding="utf-8") as f:
#         json.dump(data, f, ensure_ascii=False, indent=2)

# prune_llm_outputs("data/AdaptEval/gsm8k_augmented_dataset_cotevaluation_temp05_0_1000.json")

import json
import re
from typing import List, Dict

def prune_after_answer_sentence(text: str) -> str:
    """
    Prune everything after a sentence starting with
    'Final answer:' or 'the answer is:' (case-insensitive).
    """
    # Match "Final answer:" or "the answer is:" at the start of a sentence,
    # and include the sentence-ending punctuation
    pattern = r"(Final answer:|answer is:)[^.!?]*[.!?]"
    
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        return text[: match.end()]
    return text

ROLE_MARKERS = [
    "Human:",
    "User:",
    "Assistant:",
]

def prune_after_so_sentence(text: str) -> str:
    """
    Keep everything up to and including a sentence that begins with 'So,'
    plus any immediately following LaTeX/math blocks.
    Prune anything that comes after.
    """
    # Match a sentence starting with "So,"
    so_pattern = r"(So,[^\n.!?]*[.!?]?)"
    match = re.search(so_pattern, text, flags=re.IGNORECASE)

    if not match:
        return text

    end = match.end()

    # Allow trailing whitespace and LaTeX/math blocks
    trailing_pattern = r"""
        (?:\s*                           # whitespace
        (?:\\\[.*?\\\]                  # \[ ... \]
        |\\\(.*?\\\)                    # \( ... \)
        |\$\$.*?\$\$                    # $$ ... $$
        |\$.*?\$                        # $ ... $
        )                               # math
        )*
    """

    trailing_match = re.match(
        trailing_pattern,
        text[end:],
        flags=re.DOTALL | re.VERBOSE
    )

    if trailing_match:
        end += trailing_match.end()

    return text[:end]


def prune_after_role_marker(text: str) -> str:
    """
    Remove everything starting from dialogue role markers
    like 'Human:', 'User:', 'Assistant:'.
    """
    pattern = r"(Human|User|Assistant)\s*:"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        return text[:match.start()]
    return text

def prune_after_instruction_preamble(text: str) -> str:
    """
    Remove everything starting from instruction-style boilerplate
    such as 'You are an AI assistant...'
    """
    pattern = (
        r"\n?\s*You\s+are\s+an\s+AI\s+assistant\b"
    )
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        return text[:match.start()]
    return text

# above functions added for qwen


def prune_after_note(text: str) -> str:
    """
    Remove anything starting from 'Note:' or '(Note:' (case-insensitive).
    """
    pattern = r"(\(|\n|\s|^)\s*note\s*:"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        return text[:match.start()]
    return text


def prune_after_conclusive_answer(text: str) -> str:
    """
    Prune everything after a sentence that clearly delivers the final answer,
    e.g. 'So, to answer the question, ...'
    """

    answer_patterns = [
        r"So,\s+to\s+answer\s+the\s+question[^.!?]*[.!?]",
        r"Therefore,\s+the\s+correct\s+answer\s+is[^.!?]*[.!?]",
        r"the\s+final\s+answer\s+is[^.!?]*[.!?]",
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return text[: match.end()]

    return text

def prune_after_therefore_correct_answer(text: str) -> str:
    """
    Keep everything up to and including the sentence that begins with:
    'Therefore, the correct answer is'
    """
    pattern = (
        r"(Therefore,\s+the\s+correct\s+answer\s+is[^.!?]*[.!?])"
    )

    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        return text[: match.end()]
    return text

def prune_after_therefore_sentence(text: str) -> str:
    """
    Keep everything up to and including a sentence that begins with 'Therefore,'
    plus any immediately following LaTeX/math blocks.
    Prune anything that comes after (e.g., instruction leakage).
    """
    # Match "Therefore, ..." sentence
    therefore_pattern = r"(Therefore,[^\n.!?]*[.!?]?)"
    match = re.search(therefore_pattern, text, flags=re.IGNORECASE)

    if not match:
        return text

    end = match.end()

    # Allow trailing whitespace and LaTeX math blocks
    trailing_pattern = r"""
        (?:\s*                           # whitespace
        (?:\\\[.*?\\\]                  # \[ ... \] math block
        |\\\(.*?\\\)                    # \( ... \)
        |\$\$.*?\$\$                    # $$ ... $$
        |\$.*?\$                        # $ ... $
        )                               # math
        )*
    """

    trailing_match = re.match(
        trailing_pattern,
        text[end:],
        flags=re.DOTALL | re.VERBOSE
    )

    if trailing_match:
        end += trailing_match.end()

    return text[:end]


def prune_after_second_final_answer(text: str) -> str:
    marker = "Final Answer:"
    
    # 2. Find the index of the first occurrence
    index = text.find(marker)
    
    if index == -1:
        return text  # Return original if marker isn't found
    
    # 3. Get the text from the start up to the marker
    prefix = text[:index + len(marker)]
    
    # 4. Get the remaining text to find the "sentence after"
    remaining = text[index + len(marker):]
    
    # 5. Use regex to find the first complete sentence ending in . ! or ?
    # This looks for any characters until the first punctuation mark.
    sentence_match = re.search(r'[^.!?]*[.!?]', remaining)
    
    if sentence_match:
        return prefix + sentence_match.group(0)
    else:
        # Fallback: if no punctuation is found, just return the whole first line
        return prefix + remaining.split('\n')[0]



META_HEADERS = [
    "evaluation",
    "correctness",
    "feedback",
    "comment",
    "comments",
    "review",
    "grading",
    "assessment",
    "analysis",  
]

def prune_llm_outputs(
    json_path: str,
    stop_markers: List[str] = None,
) -> List[Dict]:
    """
    Load a JSON file (list of dicts), prune over-generated 'output' fields,
    and return the cleaned list.

    Pruning rules:
    - Remove everything starting from '\n\n#### Evaluation'
    - Remove everything starting from other stop_markers (if provided)
    """

    if stop_markers is None:
        stop_markers = [
            "### Instruction:",
            "### Response:",
        ]

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for entry in data:

        text = entry["cot_output"] if "cot_output" in entry else entry["output"]
        entry["original_output"] = text

        # Cut at known evaluation sections
        eval_match = re.search(r"\n\s*####\s*Evaluation\b", text)
        if eval_match:
            text = text[:eval_match.start()]

        # Cut at any markdown header level ≥ 3 (Final Answer, Comments, boilerplate)
        markdown_junk = re.search(r"\n\s*#{3,}\s+", text)
        if markdown_junk:
            text = text[:markdown_junk.start()]


        # 3. Existing stop markers
        cut_positions = [
            text.find(marker) for marker in stop_markers if marker in text
        ]
        if cut_positions:
            text = text[:min(cut_positions)]

        meta_pattern = (
        r"(?:\n|\s)"          # newline OR space
        r"#{3,}\s*"           # ###, ####, ######
        r"(" + "|".join(META_HEADERS) + r")\b"
        )

        meta_match = re.search(meta_pattern, text, flags=re.IGNORECASE)
        if meta_match:
            text = text[:meta_match.start()]
        text = prune_after_role_marker(text)
        text = prune_after_answer_sentence(text)
        text = prune_after_instruction_preamble(text)
        text = prune_after_therefore_correct_answer(text)
        text = prune_after_note(text)
        text = prune_after_second_final_answer(text)
        text = prune_after_conclusive_answer(text)
        text = prune_after_therefore_sentence(text)
        text = prune_after_so_sentence(text)
        
        entry["output"] = text.strip()

        

    final_path = json_path.replace(".json", "_pruned.json")
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return data
prune_llm_outputs("/Users/armin/Downloads/augmented_dataset_CoLoTa_qwen2_0_2000.json")