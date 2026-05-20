# Module 7 Week B — Integration Task: Summarization & Integrated Evaluation Report

This is the starter repo for the Module 7 Week B Integration Task. **The integrated evaluation report you produce here is the M7 deliverable.**

The full integration guide is at <a href="https://levelup-applied-ai.github.io/aispire-14005-pages/modules/module-7/496c1c2b" target="_blank">the integration guide page</a> — read it first.

## Quick start

```bash
pip install -r requirements.txt
make summarize    # runs full pipeline; first run downloads ~250 MB
```

The first call to `pipeline("summarization", ...)` downloads the model. Plan ~3 minutes for the first run; subsequent runs use cached weights. The full evaluation on 120 articles completes in ~6–8 minutes on CPU after the model is cached.

## Model

This integration uses **`sshleifer/distilbart-cnn-6-6`**, a distilled version of Facebook's BART model fine-tuned on the CNN/DailyMail news summarization dataset. It is an encoder–decoder (sequence-to-sequence) transformer that generates abstractive summaries — meaning it writes new sentences rather than copying phrases directly from the source. The model is loaded from Hugging Face Hub at runtime; no model file is committed to the repo. Summarization runs with deterministic beam search (`do_sample=False`, `num_beams=4`) and produces outputs between 30 and 120 tokens.

## Corpus

The evaluation runs on **120 tech, entertainment, and digital-culture news articles** drawn from the curated `data/tech_news_articles.csv` file, which contains 1,033 articles sourced from the [`glnmario/news-qa-summarization`](https://huggingface.co/datasets/glnmario/news-qa-summarization) dataset on Hugging Face. Reference summaries for the 120 evaluated articles are in `data/tech_news_summaries_reference.csv`; these are CNN editor-authored summaries that ship with the source dataset and serve as the gold standard for ROUGE scoring.

## Re-run command

```bash
make summarize
```

This runs `python summarize.py` against the full 120-article evaluation set and writes `summary_predictions.csv` and `summary_metrics.json` to the repo root.

## What you will produce

Committed:
- `summarize.py` — your implementation
- Updated `README.md` — this file
- `summary_predictions.csv` — 120 rows with reference, predicted, and per-summary ROUGE
- `summary_metrics.json` — aggregate ROUGE-1/2/L F1
- `integrated-evaluation-report.md` — six-section integrated report (the M7 deliverable). Includes an optional Section 7 (Challenge Extensions) for learners completing challenge tiers — see the integration's learner guide.

**No model file** — pre-trained model loads from Hugging Face Hub at runtime.

## Data

- `data/tech_news_articles.csv` — 1,033 tech / entertainment / digital-culture news articles, curated from <a href="https://huggingface.co/datasets/glnmario/news-qa-summarization" target="_blank">glnmario/news-qa-summarization</a>. The full pool is here for inspection and stretch use; the integration evaluates on the 120-article subset that has reference summaries.
- `data/tech_news_summaries_reference.csv` — 120 reference summaries (one per evaluated article), shipped with the curated dataset (CNN editor-authored summaries from the source dataset).
- `data/tiny_articles_smoke.csv` + `data/tiny_refs_smoke.csv` — 3-row CI smoke fixtures (articles and references in separate files, matching the real-data schema).

## Make targets

```bash
make summarize    # full pipeline against the 120-article evaluation set
make smoke        # CI-only target — 3-row fixture
make clean        # remove generated outputs
```

## Submission

Open a Pull Request from your working branch into `main`. The autograder runs `make smoke` against the 3-row fixture and validates artifact schemas. PR description requirements are in the integration guide.

---

## License

This repository is provided for educational use only. See [LICENSE](LICENSE) for terms.

You may clone and modify this repository for personal learning and practice, and reference code you wrote here in your professional portfolio. Redistribution outside this course is not permitted.
