"""
Autograder tests for Module 7 Week B — Integration Task.

CI runs `make smoke` first (writes summary_predictions_smoke.csv); these tests
then verify both function-level correctness and produced artifacts.
"""

import json
import math
import os
import sys

import pandas as pd
import pytest

# Add repo root to sys.path so we can import summarize.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import summarize  # noqa: E402


REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
SUBMITTED_PREDS = os.path.join(REPO_ROOT, "summary_predictions.csv")
SUBMITTED_METRICS = os.path.join(REPO_ROOT, "summary_metrics.json")
SMOKE_PREDS = os.path.join(REPO_ROOT, "summary_predictions_smoke.csv")
INTEGRATED_REPORT = os.path.join(REPO_ROOT, "integrated-evaluation-report.md")
MAKEFILE = os.path.join(REPO_ROOT, "Makefile")
README = os.path.join(REPO_ROOT, "README.md")


# -- Function-level correctness ---------------------------------------------

def test_compute_rouge_returns_three_keys():
    out = summarize.compute_rouge("hello world", "hello world")
    assert set(out.keys()) == {"rouge1", "rouge2", "rougeL"}


def test_compute_rouge_perfect_match_is_one():
    out = summarize.compute_rouge("the quick brown fox", "the quick brown fox")
    for k, v in out.items():
        assert math.isclose(v, 1.0, rel_tol=1e-6), f"{k} = {v}, expected 1.0"


def test_compute_rouge_values_in_unit_interval():
    out = summarize.compute_rouge("brown fox jumps", "the quick brown fox jumps high")
    for k, v in out.items():
        assert 0.0 <= v <= 1.0, f"{k} = {v} not in [0, 1]"


# -- Pipeline construction ---------------------------------------------------

@pytest.fixture(scope="module")
def summ_pipeline():
    return summarize.build_summarizer(summarize.get_summarizer_model_name())


def test_build_summarizer_returns_callable(summ_pipeline):
    assert callable(summ_pipeline)


def test_summarize_one_returns_string(summ_pipeline):
    text = (
        "Researchers reported a new battery technology achieving 1000 cycles "
        "with under 5% loss. The team plans commercial trials next year. "
        "The breakthrough could enable longer-range electric vehicles."
    )
    out = summarize.summarize_one(summ_pipeline, text, max_length=40, min_length=10)
    assert isinstance(out, str)
    assert len(out) > 0


# -- Evaluate harness --------------------------------------------------------

def _load_smoke_frames():
    articles_df = pd.read_csv(os.path.join(REPO_ROOT, "data", "tiny_articles_smoke.csv"))
    refs_df = pd.read_csv(os.path.join(REPO_ROOT, "data", "tiny_refs_smoke.csv"))
    return articles_df, refs_df


def test_evaluate_summaries_returns_required_keys(summ_pipeline):
    articles_df, refs_df = _load_smoke_frames()
    result = summarize.evaluate_summaries(summ_pipeline, articles_df, refs_df)
    for k in ("rouge1", "rouge2", "rougeL", "n", "predictions"):
        assert k in result, f"missing key: {k}"


def test_evaluate_summaries_predictions_schema(summ_pipeline):
    articles_df, refs_df = _load_smoke_frames()
    result = summarize.evaluate_summaries(summ_pipeline, articles_df, refs_df)
    required = {"article_id", "reference_summary", "predicted_summary", "rouge1", "rouge2", "rougeL"}
    for pred in result["predictions"]:
        assert required.issubset(pred.keys()), f"missing keys: {required - pred.keys()}"


# -- Smoke artifact tests (after `make smoke` runs in CI) -------------------

def test_smoke_predictions_generated_by_pipeline():
    if not os.path.exists(SMOKE_PREDS):
        pytest.skip("smoke predictions not produced (run `make smoke` first)")
    df = pd.read_csv(SMOKE_PREDS)
    required = {"article_id", "reference_summary", "predicted_summary", "rouge1", "rouge2", "rougeL"}
    assert required.issubset(df.columns), f"missing columns: {required - set(df.columns)}"
    for col in ("rouge1", "rouge2", "rougeL"):
        assert df[col].between(0.0, 1.0).all(), f"{col} not in [0, 1]"


# -- Submitted artifact tests (the learner's real 120-article output) -------

def test_submitted_summary_predictions_csv_columns():
    if not os.path.exists(SUBMITTED_PREDS):
        pytest.skip(f"{SUBMITTED_PREDS} not present (learner has not committed)")
    df = pd.read_csv(SUBMITTED_PREDS)
    required = {"article_id", "reference_summary", "predicted_summary", "rouge1", "rouge2", "rougeL"}
    assert required.issubset(df.columns)


def test_submitted_summary_predictions_row_count():
    if not os.path.exists(SUBMITTED_PREDS):
        pytest.skip(f"{SUBMITTED_PREDS} not present")
    df = pd.read_csv(SUBMITTED_PREDS)
    assert len(df) == 120, f"expected 120 rows, got {len(df)}"


def test_summary_metrics_json_schema():
    if not os.path.exists(SUBMITTED_METRICS):
        pytest.skip(f"{SUBMITTED_METRICS} not present")
    with open(SUBMITTED_METRICS) as f:
        metrics = json.load(f)
    for k in ("rouge1", "rouge2", "rougeL", "n", "model"):
        assert k in metrics


# -- Reproducibility scaffolding ---------------------------------------------

def test_makefile_targets_present():
    with open(MAKEFILE) as f:
        content = f.read()
    for target in ("summarize:", "smoke:", "clean:"):
        assert target in content, f"missing Makefile target: {target}"


def test_readme_updated():
    """README should be updated with model / corpus / re-run info — > starter baseline (~2,700 chars) + ~700 chars of real additions AND mention model."""
    with open(README) as f:
        content = f.read()
    # Starter README is ~2,700 chars; threshold requires substantive learner additions, not a one-word edit.
    assert len(content) >= 3400, f"README appears unmodified or only superficially edited ({len(content)} chars)"
    assert "distilbart" in content.lower() or "bart" in content.lower(), "README should mention the model"


# -- Integrated evaluation report (the M7 deliverable) ----------------------

def test_integrated_report_exists_and_has_six_sections():
    if not os.path.exists(INTEGRATED_REPORT):
        pytest.skip(f"{INTEGRATED_REPORT} not present")
    with open(INTEGRATED_REPORT) as f:
        content = f.read()
    assert len(content) >= 1500, f"report length {len(content)} < 1500 chars"
    # Six required section headers (loose match — case-insensitive)
    headers_lower = content.lower()
    required_headers = [
        "comparison table",
        "findings",
        "faithfulness",
        "production decision matrix",
        "what you would do differently",
        "limits of the evaluation",
    ]
    for h in required_headers:
        assert h in headers_lower, f"required section '{h}' not found in report"


def test_integrated_report_numbers_in_plausible_ranges():
    """Catch omissions and absurd values. NOT a fabrication check —
    the TA verifies number-vs-file consistency separately (see TA Rubric)."""
    if not os.path.exists(INTEGRATED_REPORT):
        pytest.skip(f"{INTEGRATED_REPORT} not present")
    import re

    with open(INTEGRATED_REPORT) as f:
        content = f.read()

    # Skip if the report is still the unmodified template (placeholders present).
    if "_(your number" in content.lower() or "_(your numbers" in content.lower():
        pytest.skip("integrated-evaluation-report.md still contains template placeholders; complete the report before pushing for the numeric check")

    # Section 7 (Challenge Extensions) is optional and legitimately contains
    # decimals outside [0, 1] — Tier 1 cites latency in seconds (e.g., 1.24)
    # and Tier 3 cites latency + disk_mb. Scan only Sections 1-6, which are
    # the required deliverable carrying ROUGE / F1 / EM metric values.
    content_for_scan = re.split(r"(?im)^##\s*(?:section\s+)?7\.?\s", content, maxsplit=1)[0]

    # Find decimal numbers in the required-section portion of the report
    nums = [float(m) for m in re.findall(r"\b0?\.\d{2,4}\b", content_for_scan)]
    # We expect to see at least a few numbers — comparison table values
    assert len(nums) >= 3, "report contains too few numeric values; comparison table likely empty"
    # All numbers should be plausible metric ranges
    for n in nums:
        assert 0.0 <= n <= 1.0, f"value {n} outside [0, 1] — likely a typo or units error"
