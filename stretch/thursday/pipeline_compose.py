"""
Module 7 Week B — Thursday Stretch (Honors): Summarize-then-QA.

Composes the Integration 7B summarizer with the Lab 7B QA pipeline.
Implements qa_full_article, qa_via_summary, evaluate_strategies, and main().

Strategy A — QA on the full article with overlapping chunking when the
             article exceeds the QA model's max-context window.
Strategy B — Summarize the article first (distilbart-cnn-6-6), then run
             extractive QA on the short summary.
"""

import json
import os
import sys

import pandas as pd

# ── Path setup ──────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import summarize
import qa_utils 

# ── Constants ────────────────────────────────────────────────────────────────
_CHUNK_SIZE = 384       # max tokens per QA window (model hard limit is ~512)
_CHUNK_OVERLAP = 64     # overlap between adjacent windows (~16 % of window)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _word_tokenize(text: str) -> list[str]:
    """Split text into whitespace-delimited tokens (cheap, no dependency)."""
    return text.split()


def _tokens_to_text(tokens: list[str]) -> str:
    return " ".join(tokens)


def _chunk_article(article: str, max_chunk: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    """
    Split *article* into overlapping windows of at most *max_chunk* tokens.

    Each window except the last has exactly *max_chunk* tokens; the final
    window may be shorter.  Adjacent windows share *overlap* tokens so that
    an answer straddling a boundary is captured by at least one window.
    """
    tokens = _word_tokenize(article)
    stride = max_chunk - overlap  # non-overlapping step size

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_chunk, len(tokens))
        chunks.append(_tokens_to_text(tokens[start:end]))
        if end == len(tokens):
            break
        start += stride

    return chunks


# ── Strategy A ───────────────────────────────────────────────────────────────

def qa_full_article(qa, question: str, article: str, max_chunk: int = _CHUNK_SIZE) -> str:
    """
    Run QA over the full article, chunking with overlap when it exceeds max_chunk tokens.

    Algorithm
    ---------
    1. Count (approximate) tokens in the article.
    2. If the article fits within max_chunk, call predict_one once.
    3. Otherwise split into overlapping windows (max_chunk tokens, 64-token overlap).
    4. Run predict_one on every window; collect (answer, score) pairs.
    5. Return the answer string from the highest-scoring window.

    Returns
    -------
    str
        The best answer span across all chunks.
    """
    tokens = _word_tokenize(article)

    # ── Case 1: fits in a single window ──────────────────────────────────────
    if len(tokens) <= max_chunk:
        result = qa_utils.predict_one(qa, question, article)
        return result["answer"]

    # ── Case 2: split into overlapping windows ────────────────────────────────
    chunks = _chunk_article(article, max_chunk=max_chunk, overlap=_CHUNK_OVERLAP)

    best_answer = ""
    best_score = -1.0

    for chunk in chunks:
        result = qa_utils.predict_one(qa, question, chunk)
        if result["score"] > best_score:
            best_score = result["score"]
            best_answer = result["answer"]

    return best_answer


# ── Strategy B ───────────────────────────────────────────────────────────────

def qa_via_summary(qa, summ, question: str, article: str, max_summary_length: int = 120) -> str:
    """
    Summarize the article first, then run QA on the summary.

    Algorithm
    ---------
    1. Call summarize.summarize_one with do_sample=False, num_beams=4
       (deterministic beam-search; matches Integration 7B defaults).
    2. Run QA on the resulting summary string with qa_utils.predict_one.
    3. Return the answer string.

    NOTE: The autograder verifies that *summ* is called before *qa* in
    this function body — do not reorder the two calls.

    Returns
    -------
    str
        The answer span extracted from the summary.
    """
    # Step 1 — summarise (summ MUST be invoked before qa)
    summary = summarize.summarize_one(
        summ,
        article,
        max_length=max_summary_length,
        min_length=30,
    )

    # Step 2 — QA on summary
    result = qa_utils.predict_one(qa, question, summary)

    return result["answer"]


# ── Evaluation harness ───────────────────────────────────────────────────────

def evaluate_strategies(qa, summ, test_set: pd.DataFrame, articles_df: pd.DataFrame) -> dict:
    """
    Run both strategies on every row of the test set; compute per-strategy EM/F1.

    Parameters
    ----------
    qa          : HuggingFace QA pipeline (from qa_utils.build_qa_pipeline)
    summ        : HuggingFace summarization pipeline (from summarize.build_summarizer)
    test_set    : DataFrame with columns [qid, article_id, question, gold_answer, …]
    articles_df : DataFrame with columns [article_id, text, …]

    Returns
    -------
    dict with keys:
        "strategy_a"  : {"em": float, "f1": float, "n": int}
        "strategy_b"  : {"em": float, "f1": float, "n": int}
        "predictions" : list of per-row dicts
    """
    predictions = []

    a_em_list, a_f1_list = [], []
    b_em_list, b_f1_list = [], []

    for _, row in test_set.iterrows():
        # ── Look up the article ───────────────────────────────────────────────
        article_rows = articles_df[articles_df["article_id"] == row["article_id"]]
        if article_rows.empty:
            print(f"[WARN] article_id {row['article_id']} not found; skipping {row['qid']}")
            continue

        article = article_rows.iloc[0]["text"]
        question = row["question"]
        gold = row["gold_answer"]

        # ── Strategy A: QA on full article (with chunking) ────────────────────
        pred_a = qa_full_article(qa, question, article)

        # ── Strategy B: summarise-then-QA ─────────────────────────────────────
        pred_b = qa_via_summary(qa, summ, question, article)

        # ── Score ─────────────────────────────────────────────────────────────
        em_a = qa_utils.exact_match(pred_a, gold)
        f1_a = qa_utils.token_f1(pred_a, gold)

        em_b = qa_utils.exact_match(pred_b, gold)
        f1_b = qa_utils.token_f1(pred_b, gold)

        a_em_list.append(em_a)
        a_f1_list.append(f1_a)
        b_em_list.append(em_b)
        b_f1_list.append(f1_b)

        predictions.append(
            {
                "qid": row["qid"],
                "question": question,
                "strategy_a_pred": pred_a,
                "strategy_b_pred": pred_b,
                "gold_answer": gold,
                "strategy_a_em": em_a,
                "strategy_a_f1": round(f1_a, 4),
                "strategy_b_em": em_b,
                "strategy_b_f1": round(f1_b, 4),
            }
        )

        print(
            f"  {row['qid']} | A-EM={em_a} A-F1={f1_a:.2f} | B-EM={em_b} B-F1={f1_b:.2f}"
            f" | gold='{gold}' | pred_a='{pred_a}' | pred_b='{pred_b}'"
        )

    n = len(predictions)
    if n == 0:
        raise RuntimeError("No predictions produced — check article_id alignment.")

    return {
        "strategy_a": {
            "em": sum(a_em_list) / n,
            "f1": sum(a_f1_list) / n,
            "n": n,
        },
        "strategy_b": {
            "em": sum(b_em_list) / n,
            "f1": sum(b_f1_list) / n,
            "n": n,
        },
        "predictions": predictions,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Load test set + articles, build pipelines, run both strategies, write artifacts."""
    test_set_path    = os.environ.get("TEST_SET_PATH",   os.path.join(HERE, "qa_test_set.csv"))
    articles_path    = os.environ.get("ARTICLES_PATH",   os.path.join(ROOT, "data", "tech_news_articles.csv"))
    out_preds_path   = os.environ.get("PREDS_OUT_PATH",  os.path.join(HERE, "compose_predictions.csv"))
    out_metrics_path = os.environ.get("METRICS_OUT_PATH", os.path.join(HERE, "compose_metrics.json"))

    print(f"Loading test set from  : {test_set_path}")
    print(f"Loading articles from  : {articles_path}")

    test_set    = pd.read_csv(test_set_path)
    articles_df = pd.read_csv(articles_path)

    print(f"\nTest set : {len(test_set)} rows")
    print(f"Articles : {len(articles_df)} rows\n")

    # Build pipelines
    print("Building QA pipeline …")
    qa = qa_utils.build_qa_pipeline(qa_utils.get_qa_model_name())

    print("Building summarizer pipeline …")
    summ = summarize.build_summarizer(summarize.get_summarizer_model_name())

    print("\nRunning evaluation …\n")
    result = evaluate_strategies(qa, summ, test_set, articles_df)

    # ── Write artifacts ───────────────────────────────────────────────────────
    pred_df = pd.DataFrame(result["predictions"])
    pred_df.to_csv(out_preds_path, index=False)
    print(f"\nPredictions written to : {out_preds_path}")

    metrics = {
        "strategy_a": result["strategy_a"],
        "strategy_b": result["strategy_b"],
        "qa_model": qa_utils.get_qa_model_name(),
        "summarizer_model": summarize.get_summarizer_model_name(),
    }
    with open(out_metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics written to     : {out_metrics_path}")

    print(f"\nStrategy A (full-article QA) — EM={result['strategy_a']['em']:.4f}, F1={result['strategy_a']['f1']:.4f}")
    print(f"Strategy B (summarize-then-QA) — EM={result['strategy_b']['em']:.4f}, F1={result['strategy_b']['f1']:.4f}")


if __name__ == "__main__":
    main()