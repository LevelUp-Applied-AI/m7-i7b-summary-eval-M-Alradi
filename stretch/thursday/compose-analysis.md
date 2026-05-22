# Summarize-then-QA — Trade-Off Analysis Memo

## 1. Test Set Design

* Total questions: 20

* Article types chosen:
  The dataset contains a mix of **medium to long-form articles**, with dense factual content (news, reports, technical explanations, and case studies). These were selected to test whether summarization preserves fine-grained details needed for QA.

* Question types:

  * Factual entity extraction (Q01, Q02, Q04, Q14, Q18)
  * Numeric extraction (Q03, Q05, Q06, Q07, Q08, Q09, Q10, Q11, Q13, Q16, Q19, Q20)
  * Causal / explanatory questions (Q12, Q17)
  * Multi-constraint / deep reasoning (Q15, Q16, Q17)

* Why these choices:
  The design stresses both **surface-level retrieval** and **deep document reasoning**, ensuring that models must either precisely locate spans (Strategy A) or rely on compressed representations (Strategy B). This allows evaluation of how summarization impacts fidelity for numeric and low-redundancy facts.

---

## 2. Strategy A Results — QA on the Full Article (with Chunking)

* Aggregate EM: **0.7000**
* Aggregate F1: **0.8496**

### Where Strategy A wins

Strategy A performs strongly on questions requiring **exact span extraction from long documents**:

* **Q03 (10,000 fixed telephones)** → exact numeric match preserved in chunk
* **Q07 (97 percent revenue)** → correct extraction despite distractor context
* **Q11 (90 minutes Twitter outage)** → temporal fact correctly retrieved

**Why it wins:** Chunking preserves local context, allowing the QA model to directly extract exact spans even in long documents.

---

### Where Strategy A loses

* **Q08 ($50 a head annually)** → predicted only “$50” (lost qualifier)
* **Q15 (injuries description)** → partial span mismatch (“skull fractures and facial cuts”)

**Why it loses:** Chunk boundaries and partial context extraction cause **loss of modifiers and qualifiers**, especially for long descriptive answers.

---

## 3. Strategy B Results — QA on the Summary

* Aggregate EM: **0.1000**
* Aggregate F1: **0.1077**

### Where Strategy B wins

* **Q04 (Barcelona)** → correct location preserved in summary
* **Q09 (nearly 3,200)** → summary retained numeric estimate correctly

**Why it wins:** When the summary preserves key named entities or headline numbers, QA succeeds even without full article context.

---

### Where Strategy B loses

* **Q01 (2006)** → summary replaced answer with unrelated general statement
* **Q07 (97 percent)** → summary replaced numeric fact with “billions”
* **Q11 (90 minutes)** → entity drift (“Twitter” instead of duration)
* **Q16, Q17** → critical numeric and causal details dropped entirely

**Why it loses:** Summarization aggressively compresses content, often **dropping low-salience but QA-critical facts** such as numbers, durations, and named entities.

---

## 4. Faithfulness Analysis (Strategy B)

**Example failure case (information loss in summarization):**

> **Article (excerpt):**
> “According to the Health Ministry, 16 people died in the clashes, including 14 from gunshots during the incident.”
>
> **Summary:**
> “Azza Hilal Suleiman was involved in protests and subsequent events escalated.”
>
> **Question:**
> How many people died according to the Health Ministry, and how many were by gunshots?
>
> **Strategy B prediction:** Azza Hilal Suleiman
> **Gold:** 16 people died, including 14 by gunshots

**What was lost in summarization:**
The summary completely removed **numerical mortality statistics and cause-of-death breakdown**, which are essential for answering the question. This demonstrates that abstractive summarization prioritizes narrative flow over factual completeness.

---

## 5. Recommendation

| Use Strategy A when…                                                      | Use Strategy B when…                                           |
| ------------------------------------------------------------------------- | -------------------------------------------------------------- |
| Questions require **exact numeric/entity extraction** (EM-critical tasks) | Questions are **high-level or thematic** (topic understanding) |
| Articles are **medium to long (multi-paragraph, fact-dense)**             | Articles are **short or highly redundant**                     |
| Expected answer is a **verbatim span (dates, numbers, names)**            | Expected answer is **semantic or descriptive**                 |

### Justification

Strategy A significantly outperforms Strategy B (**EM: 0.70 vs 0.10; F1: 0.85 vs 0.11**), showing that summarization introduces severe loss of answer-critical details. However, Strategy B may still be useful for reducing compute cost in cases where **precision is not required and approximate understanding is sufficient**.

---