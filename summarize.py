"""
Module 7 Week B — Integration Task: Summarization & Integrated Evaluation Report.

Implement the functions below. See the integration guide for full task descriptions.

The integrated evaluation report (`integrated-evaluation-report.md`) is the M7
deliverable. Write it by hand based on the metrics produced by main().
"""

import json
import os

import pandas as pd


# -- Helpers (provided — do NOT modify) --------------------------------------

def get_summarizer_model_name() -> str:
    """Return env override (CI smoke) or the default summarization model."""
    return os.environ.get("SUMM_MODEL_FOR_CI", "sshleifer/distilbart-cnn-6-6")


def _articles_path() -> str:
    return os.environ.get("ARTICLES_PATH", "data/tech_news_articles.csv")


def _references_path() -> str:
    return os.environ.get("REFERENCES_PATH", "data/tech_news_summaries_reference.csv")


def _output_path() -> str:
    return os.environ.get("OUTPUT_PATH", "summary_predictions.csv")


# -- Task 1: Pipeline + single-document summarization ------------------------

def build_summarizer(model_name: str):
    """Construct a Hugging Face summarization pipeline."""
    # TODO: build a summarization pipeline using the given model name (same as the drill)
    raise NotImplementedError("build_summarizer not implemented")


def summarize_one(summ, text: str, max_length: int = 120, min_length: int = 30) -> str:
    """
    Summarize one document with deterministic beam search.

    Use do_sample=False, num_beams=4. Return the summary STRING from
    [0]["summary_text"].
    """
    # TODO: invoke the pipeline with deterministic generation parameters (no sampling, beam search) and return the summary string
    raise NotImplementedError("summarize_one not implemented")


# -- Task 2: ROUGE -----------------------------------------------------------

def compute_rouge(pred: str, ref: str) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L F1.

    Use rouge_score.rouge_scorer.RougeScorer with use_stemmer=True.
    Argument order: scorer.score(reference, predicted) — REFERENCE FIRST.

    Returns {"rouge1": float, "rouge2": float, "rougeL": float}, all F1.
    """
    # TODO: build a stemming-enabled ROUGE scorer over the three metric variants
    # TODO: score the (reference, predicted) pair and return F1 measures only (note argument order)
    raise NotImplementedError("compute_rouge not implemented")


# -- Task 3: Evaluate over the corpus ----------------------------------------

def evaluate_summaries(summ, articles_df: pd.DataFrame, refs_df: pd.DataFrame) -> dict:
    """
    Summarize each article and score against its reference.

    Returns:
        {
          "rouge1": float, "rouge2": float, "rougeL": float,
          "n": int,
          "predictions": [
            {article_id, reference_summary, predicted_summary, rouge1, rouge2, rougeL},
            ...
          ],
        }

    Joins articles_df and refs_df on article_id.
    """
    # TODO: merge the two DataFrames on article_id
    # TODO: iterate, summarize each article, compute ROUGE vs. reference
    # TODO: aggregate (mean across summaries) and return the dict
    raise NotImplementedError("evaluate_summaries not implemented")


# -- Task 4: Orchestrate -----------------------------------------------------

def main() -> None:
    """Load data, build pipeline, evaluate, write artifacts."""
    articles_df = pd.read_csv(_articles_path())
    refs_df = pd.read_csv(_references_path())

    summ = build_summarizer(get_summarizer_model_name())
    result = evaluate_summaries(summ, articles_df, refs_df)

    # Write predictions CSV
    pred_df = pd.DataFrame(result["predictions"])
    pred_df.to_csv(_output_path(), index=False)

    # Write metrics JSON
    metrics = {
        "rouge1": result["rouge1"],
        "rouge2": result["rouge2"],
        "rougeL": result["rougeL"],
        "n": result["n"],
        "model": get_summarizer_model_name(),
    }
    metrics_path = _output_path().replace("predictions", "metrics").replace(".csv", ".json")
    if metrics_path == _output_path():  # safety: ensure rename happened
        metrics_path = "summary_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"ROUGE-1 = {result['rouge1']:.4f}")
    print(f"ROUGE-2 = {result['rouge2']:.4f}")
    print(f"ROUGE-L = {result['rougeL']:.4f}")
    print(f"n = {result['n']}")


if __name__ == "__main__":
    main()
