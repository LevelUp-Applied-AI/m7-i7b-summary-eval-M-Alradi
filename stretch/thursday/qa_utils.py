"""
Module 7 Week B — Thursday Stretch: QA Utilities.

Provides: build_qa_pipeline, predict_one, exact_match, token_f1,
          normalize_answer, and get_qa_model_name.

Model: deepset/roberta-base-squad2  (extractive QA on SQuAD 2.0)
"""

import os
import re
import string
import collections

from transformers import pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_qa_model_name() -> str:
    """Return env override (CI smoke) or the default extractive QA model."""
    return os.environ.get("QA_MODEL_FOR_CI", "deepset/roberta-base-squad2")


# ---------------------------------------------------------------------------
# Task 1: Pipeline + single-document QA
# ---------------------------------------------------------------------------

def build_qa_pipeline(model_name: str):
    """Construct a Hugging Face extractive question-answering pipeline."""
    return pipeline(
        "question-answering",
        model=model_name,
        tokenizer=model_name,
    )


def predict_one(qa, question: str, context: str) -> dict:
    """
    Run extractive QA on a single (question, context) pair.

    Returns the pipeline's raw result dict, which includes at minimum:
        {"answer": str, "score": float, "start": int, "end": int}
    """
    result = qa({"question": question, "context": context})
    return result


# ---------------------------------------------------------------------------
# Task 2: EM / F1 metrics  (SQuAD-style evaluation)
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """
    Lowercase, remove punctuation, articles, and extra whitespace.

    This is the canonical SQuAD normalization used before EM/F1 computation.
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(prediction: str, gold: str) -> int:
    """
    Return 1 if the normalized prediction equals the normalized gold, else 0.
    """
    return int(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    """
    Compute token-level F1 between prediction and gold (SQuAD-style).

    Tokenizes both strings on whitespace after normalization, then computes
    precision / recall over the token bag-of-words overlap.
    Returns a float in [0, 1].
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1