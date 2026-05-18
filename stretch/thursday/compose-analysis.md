# Summarize-then-QA — Trade-Off Analysis Memo

> Replace each placeholder section. Memo target: ~1.5 pages. The TA rubric rewards quantitative thresholds and concrete examples.

## 1. Test Set Design

- Total questions: 20
- Article types chosen: _(short / medium / long; which lengths and why)_
- Question types: _(factual, entity-attribution, causal, top-of-document, deep-in-document — list your distribution)_
- Why these choices: _(one paragraph)_

## 2. Strategy A Results — QA on the Full Article (with Chunking)

- Aggregate EM: _(value)_; Aggregate F1: _(value)_
- Where Strategy A wins: _(name 2–3 (qid)s and explain why; cite articles whose answer is deep in the document)_
- Where Strategy A loses: _(name 1–2 (qid)s and explain — chunking noise, distractor entities in irrelevant chunks, etc.)_

## 3. Strategy B Results — QA on the Summary

- Aggregate EM: _(value)_; Aggregate F1: _(value)_
- Where Strategy B wins: _(name 2–3 (qid)s; typically top-of-document or high-level questions)_
- Where Strategy B loses: _(name 1–2 (qid)s; typically when summarization omitted the answer evidence)_

## 4. Faithfulness Analysis (Strategy B)

**Required:** at least one example where the summary omitted the evidence Strategy B needed to answer correctly.

> Article (excerpt): _(quote)_
>
> Summary: _(quote)_
>
> Question: _(your question)_
> Strategy B prediction: _(prediction)_
> Gold: _(gold)_
>
> What was lost in summarization: _(1–2 sentences — the specific information the summary omitted that the QA model needed)_

## 5. Recommendation

Specify quantitative thresholds for when to use each strategy. Anchor in your measured numbers.

| Use Strategy A when… | Use Strategy B when… |
|---|---|
| _(quantitative threshold tied to article length, question depth, etc.)_ | _(quantitative threshold)_ |

Justification: _(1–2 sentences explaining the trade-off — faithfulness loss vs. compute savings vs. answer accuracy. Reference your measured numbers.)_
