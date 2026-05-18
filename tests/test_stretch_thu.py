"""
Autograder tests for the Thursday Honors stretch — Summarize-then-QA.

Triggered by .github/workflows/stretch-thu-autograder.yml on stretch-thu-*
branches (and PRs to main that touch stretch/thursday/**). Skips cleanly when
the stretch directory hasn't been populated.
"""

import ast
import json
import os
import sys

import pandas as pd
import pytest

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
STRETCH_DIR = os.path.join(REPO_ROOT, "stretch", "thursday")
QA_TEST_SET = os.path.join(STRETCH_DIR, "qa_test_set.csv")
PIPELINE_COMPOSE = os.path.join(STRETCH_DIR, "pipeline_compose.py")
QA_UTILS = os.path.join(STRETCH_DIR, "qa_utils.py")
COMPOSE_PREDS = os.path.join(STRETCH_DIR, "compose_predictions.csv")
COMPOSE_METRICS = os.path.join(STRETCH_DIR, "compose_metrics.json")
COMPOSE_ANALYSIS = os.path.join(STRETCH_DIR, "compose-analysis.md")
ARTICLES_CSV = os.path.join(REPO_ROOT, "data", "tech_news_articles.csv")


def _stretch_started() -> bool:
    if not os.path.exists(QA_TEST_SET):
        return False
    df = pd.read_csv(QA_TEST_SET)
    if len(df) == 0:
        return False
    return not df["question"].astype(str).str.contains("REPLACE_WITH", na=False).all()


# -- qa_test_set.csv schema + content ----------------------------------------

def test_qa_test_set_has_required_columns():
    if not _stretch_started():
        pytest.skip("Stretch not started — qa_test_set.csv still has placeholder rows")
    df = pd.read_csv(QA_TEST_SET)
    required = {"qid", "article_id", "question", "gold_answer", "gold_evidence_in_article"}
    assert required.issubset(df.columns), f"missing columns: {required - set(df.columns)}"


def test_qa_test_set_row_count():
    if not _stretch_started():
        pytest.skip("Stretch not started")
    df = pd.read_csv(QA_TEST_SET)
    assert len(df) == 20, f"expected 20 rows, got {len(df)}"


def test_qa_test_set_gold_answers_in_articles():
    if not _stretch_started():
        pytest.skip("Stretch not started")
    qa_df = pd.read_csv(QA_TEST_SET)
    art_df = pd.read_csv(ARTICLES_CSV)
    art_lookup = dict(zip(art_df["article_id"].astype(str), art_df["text"].astype(str)))
    for _, row in qa_df.iterrows():
        art_id = str(row["article_id"])
        assert art_id in art_lookup, f"qid={row['qid']}: article_id {art_id} not in tech_news_articles.csv"
        article_text = art_lookup[art_id].lower()
        gold = str(row["gold_answer"]).lower()
        assert gold in article_text, (
            f"qid={row['qid']}: gold answer {row['gold_answer']!r} not a substring "
            f"of article {art_id} (extractive QA constraint)"
        )


# -- qa_utils.py was provided by the learner ---------------------------------

def test_qa_utils_provided():
    if not _stretch_started():
        pytest.skip("Stretch not started")
    assert os.path.exists(QA_UTILS), (
        f"{QA_UTILS} not found — copy build_qa_pipeline, predict_one, exact_match, "
        f"token_f1, normalize_answer, get_qa_model_name from your Lab 7B lab.py"
    )
    src = open(QA_UTILS).read()
    required = ["build_qa_pipeline", "predict_one", "exact_match", "token_f1", "normalize_answer", "get_qa_model_name"]
    missing = [name for name in required if f"def {name}" not in src]
    assert not missing, f"qa_utils.py missing required functions: {missing}"


# -- pipeline_compose.py structural checks -----------------------------------

def test_qa_via_summary_actually_uses_summarizer():
    """
    Behavioral check: when qa_via_summary runs, the summarizer is actually invoked.

    Uses a spy summarizer that records invocations. The check passes whether the
    learner uses a helper, a wrapper, a different control flow, or anything else
    — as long as the summarizer participates in the path that produces the answer.
    A correct full-article-only implementation (no summarization) fails this check;
    that's the entire point of the assignment.
    """
    if not _stretch_started():
        pytest.skip("Stretch not started")
    sys.path.insert(0, REPO_ROOT)
    sys.path.insert(0, STRETCH_DIR)

    import pipeline_compose  # noqa: E402
    from unittest.mock import MagicMock

    # Spy summarizer: records every call, returns a fixed short string so the
    # downstream QA call has SOMETHING to read.
    spy_summ = MagicMock(return_value=[{"summary_text": "Stanford reports new battery technology."}])

    # Spy QA pipeline: returns a span from whatever context it receives, so we
    # can also confirm it was called on the summary (not the full article).
    def spy_qa(question=None, context=None, **kwargs):
        # Return a real-shape pipeline result; the answer doesn't matter for this test
        return {"answer": context.split()[0] if context else "", "score": 0.9, "start": 0, "end": 0}
    spy_qa = MagicMock(side_effect=spy_qa)

    article = (
        "Stanford researchers announced a new battery technology this week. "
        "The breakthrough achieved 1000 charge cycles with under 5 percent capacity loss "
        "in laboratory tests. The team plans to license the design to a manufacturer "
        "for commercial trials in 2027."
    )

    # Call the learner's function. If they wrote it correctly (or via any helper /
    # wrapper / structure), spy_summ gets called at least once.
    pipeline_compose.qa_via_summary(spy_qa, spy_summ, "What technology was announced?", article)

    assert spy_summ.call_count >= 1, (
        "qa_via_summary did not invoke the summarizer. Strategy B requires summarizing "
        "the article before running QA — otherwise it's the same as Strategy A."
    )


def test_qa_full_article_and_via_summary_signatures():
    """Both strategy functions must be defined."""
    if not _stretch_started():
        pytest.skip("Stretch not started")
    src = open(PIPELINE_COMPOSE).read()
    assert "def qa_full_article" in src, "qa_full_article not defined"
    assert "def qa_via_summary" in src, "qa_via_summary not defined"


# -- evaluate_strategies returns per-strategy structure ----------------------

def test_evaluate_strategies_returns_per_strategy_breakdown():
    """AST + docstring check on evaluate_strategies."""
    if not _stretch_started():
        pytest.skip("Stretch not started")
    src = open(PIPELINE_COMPOSE).read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "evaluate_strategies":
            doc = ast.get_docstring(node) or ""
            assert "strategy_a" in doc and "strategy_b" in doc, (
                "evaluate_strategies docstring must describe strategy_a and strategy_b sub-dicts"
            )
            return
    pytest.fail("evaluate_strategies not defined")


# -- Output artifact tests ---------------------------------------------------

def test_compose_predictions_csv_columns():
    if not os.path.exists(COMPOSE_PREDS):
        pytest.skip(f"{COMPOSE_PREDS} not present — run pipeline_compose.py first")
    df = pd.read_csv(COMPOSE_PREDS)
    required = {
        "qid", "question", "strategy_a_pred", "strategy_b_pred", "gold_answer",
        "strategy_a_em", "strategy_a_f1", "strategy_b_em", "strategy_b_f1",
    }
    assert required.issubset(df.columns), f"missing columns: {required - set(df.columns)}"


def test_compose_metrics_json_schema():
    if not os.path.exists(COMPOSE_METRICS):
        pytest.skip(f"{COMPOSE_METRICS} not present")
    with open(COMPOSE_METRICS) as f:
        m = json.load(f)
    assert "strategy_a" in m and "strategy_b" in m, "missing strategy_a or strategy_b"
    for k in ("em", "f1", "n"):
        assert k in m["strategy_a"], f"strategy_a missing {k}"
        assert k in m["strategy_b"], f"strategy_b missing {k}"


def test_compose_analysis_has_required_sections():
    if not os.path.exists(COMPOSE_ANALYSIS):
        pytest.skip(f"{COMPOSE_ANALYSIS} not present")
    content = open(COMPOSE_ANALYSIS).read()
    lower = content.lower()
    required = ["test set design", "strategy a", "strategy b", "faithfulness", "recommendation"]
    missing = [s for s in required if s not in lower]
    assert not missing, f"missing required sections: {missing}"

    # Catch unmodified template
    if "_(value)_" in content or "_(quote)_" in content:
        pytest.fail("compose-analysis.md still contains template placeholders — replace them")
