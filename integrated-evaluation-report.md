# Integrated Evaluation Report — Module 7

This report synthesizes measurements from three weeks of applied NLP work:
Lab 7A (fine-tuned sentiment classification), Lab 7B (pre-trained extractive QA), and Integration 7B (pre-trained abstractive summarization).

---

## Section 1: Comparison Table

| Task | Approach | Model | Training cost | Inference cost | Quality metric | Value |
|---|---|---|---|---|---|---|
| Sentiment classification (Lab 7A) | Fine-tuning | DistilBERT | ~30 min CPU + 3,000 labels | ~50 ms / example | Macro-F1 | 0.87 |
| Domain transfer of fine-tuned classifier (Integration 7A) | Fine-tuned model on out-of-domain data | DistilBERT (same) | Already trained | ~50 ms / example | Domain-shift judgment | Noticeable degradation — app-review vocabulary does not transfer well to tech news phrasing |
| Extractive QA (Lab 7B) | Pre-trained inference | distilbert-base-cased-distilled-squad | 0 | ~50 ms / example | EM / Token-F1 | EM: 0.41 / F1: 0.58 |
| Abstractive summarization (Integration 7B) | Pre-trained inference | distilgitbart-cnn-6-6 | 0 | ~3 sec / example | ROUGE-1 / ROUGE-2 / ROUGE-L F1 | 0.368 / 0.157 / 0.267 |

---

## Section 2: Findings

- **Fine-tuning earns its cost when the domain is stable.** The DistilBERT classifier reached 0.87 macro-F1 on app-review sentiment, which is solid for a CPU-only run. That performance only holds inside the training distribution — when we applied the same model to tech news articles (Integration 7A), the predicted labels became unreliable, because the vocabulary and sentence structure are meaningfully different. Fine-tuning buys a tight fit to one domain, not a portable capability.

- **Pre-trained QA works when the answer is literally in the passage.** On the curated tech-news QA set, the distilbert-base-cased-distilled-squad model achieved an exact match of 0.41 and token-F1 of 0.58. The gap between those two numbers tells the story: the model often finds the right neighborhood but extracts a span that is slightly too long or starts a word early, which tanks EM while still scoring partial F1 credit. Questions requiring any inference beyond span lookup were essentially unsolvable.

- **Pre-trained summarization is cheap and surprisingly reasonable.** ROUGE-1 of 0.37 and ROUGE-L of 0.27 on 120 articles, with zero training cost and no labeled data, is a useful baseline for a news domain. The model tends to front-load the most newsworthy sentence and stitch in supporting detail, which aligns well with how the CNN reference summaries are written.

- **ROUGE scores compress a lot of variation.** Across the 120 articles, individual ROUGE-L scores ranged from under 0.11 (nearly useless) to over 0.60 (near-perfect overlap). An aggregate of 0.27 hides the fact that on some articles the model produces an excellent summary and on others it picks up the wrong thread entirely. Mean ROUGE alone is not enough to characterize system quality.

- **Domain match matters for both approaches.** Both the fine-tuned classifier and the pre-trained summarizer were trained primarily on CNN/DailyMail-style content. The summarizer happens to be evaluated on that same style of content (tech news from the same source distribution), which partly explains why ROUGE scores are reasonable. The classifier was trained on app reviews, a completely different register, which explains why its domain transfer failed.

---

## Section 3: Faithfulness Check

Below are three summaries drawn from `summary_predictions.csv` — one from the high end of ROUGE-L, one from the middle, and one from the low end. For each, I note whether every claim in the predicted summary is actually supported by the article.

---

**High ROUGE-L — NEWS_0068 (ROUGE-1: 0.549, ROUGE-L: 0.549)**

*Reference summary:*
Jack Wagner discovered earlier this month that he has a long-lost 23-year-old daughter. The girl, named Carrie, introduced herself to Wagner backstage at a concert. Reports say Carrie, who was given up for adoption as an infant.

*Predicted summary:*
Jack Wagner, 52, discovered earlier this month that he has a long-lost daughter from a "fling" he had with the girl's mother in the late 1980s. The girl, named Carrie, introduced herself to Wagner backstage at a November 4 concert in Boca Raton, Florida. Less than two weeks after...

*Faithfulness assessment:* **Faithful.** The predicted summary sticks closely to verifiable facts in the article — the age, the name, the location and date of the concert. High ROUGE here reflects genuine lexical overlap and factual accuracy working together. This is the best-case scenario where ROUGE is actually a reliable signal.

---

**Mid ROUGE-L — NEWS_0040 (ROUGE-1: 0.367, ROUGE-L: 0.245)**

*Reference summary:*
Google announces Google Voice, a phone service over the Internet. The service offers "speech-to-text" transcriptions of voice messages. Head of Google Voice: company plans to offer service for free. Public users sign up for service starting in "a number of weeks."

*Predicted summary:*
Google Voice is a new version of the GrandCentral technology Google acquired in July 2007. The new version could let people centralize telephone services, screen their calls, and listen to voice mail over the Web. Google now uses its speech-to-text technology to transcribe voice mail, making it possible to read messages...

*Faithfulness assessment:* **Faithful, but emphasizes different things.** The predicted summary leads with the GrandCentral acquisition history, while the reference leads with the announcement and free-pricing angle. All claims in the prediction are supportable from the article. ROUGE sits in the middle because the two summaries prioritize different aspects of the same story — a limitation ROUGE cannot distinguish from actual error. A reader would find the prediction perfectly useful even though it scores mid-range.

---

**Low ROUGE-L — NEWS_0033 (ROUGE-1: 0.212, ROUGE-L: 0.118)**

*Reference summary:*
James Cameron's "Avatar" had its world premiere in London yesterday. The 3D sci-fi epic goes on public release worldwide on December 18. Read what the critics have said so far.

*Predicted summary:*
"Avatar" uses tailor-made technology to create the most astonishing visual effects yet seen on screen. It's an unprecedented marriage of technology and storytelling which is on the whole remarkably successful. The state-of-the-art 3D technology draws us in, but it is the vivid weirdness of Cameron's...

*Faithfulness assessment:* **Faithful to the article content, but the reference is a news brief; the prediction is a review excerpt.** The reference summary is a short, factual event announcement (premiere date, release date). The model instead pulled language that reads like critical assessment of the film. Both are technically grounded in the source text — the article covers both the premiere logistics and critical reception. Nothing in the prediction is hallucinated, but ROUGE tanks because the word overlap with the event-style reference is near zero. This is a clean example of ROUGE penalizing a style mismatch rather than a factual error — the kind of failure mode that makes low ROUGE scores hard to interpret in isolation.

---

## Section 4: Production Decision Matrix

| Scenario | Recommendation | Justification |
|---|---|---|
| Real-time app-review sentiment dashboard for a trading desk | **Fine-tuning** | The 0.87 macro-F1 achieved with 3,000 labeled app reviews demonstrates that fine-tuning on in-domain data produces the reliability a live trading tool requires; pre-trained inference on this task (without labeled app-review data) would have no equivalent quality guarantee. |
| Internal tech/entertainment news summary digest for a newsroom team | **Pre-trained inference** | The distilbart-cnn-6-6 model was itself trained on CNN/DailyMail news and reached ROUGE-1 of 0.37 on our tech news corpus at zero additional training cost, making it a reasonable starting point; the newsroom context also tolerates summaries that capture the gist rather than match a gold standard word-for-word. |
| Domain-expert QA on legal contracts | **Fine-tuning** | The pre-trained QA model hit only 0.41 exact match on relatively clean tech news passages; legal contracts use specialized vocabulary and require precise span extraction where an out-of-domain model would perform worse, making a fine-tuned model on labeled contract QA pairs the right investment despite the cost. |

---

## Section 5: What You Would Do Differently

If we had a labeled summarization dataset for the tech and entertainment news domain — even a few hundred human-written reference summaries beyond what already ships with this corpus — the most valuable single investment would be fine-tuning the summarizer on that data. Right now, distilbart-cnn-6-6 is a CNN/DailyMail model applied to a broadly similar but not identical domain. The low-ROUGE cases (like the Avatar example in Section 3) suggest the model sometimes latches onto the wrong part of an article because its internal sense of "what matters" was learned from a slightly different editorial style. Supervised fine-tuning on even a small in-domain set would likely shift those low-ROUGE outliers upward more than it would improve the already-good high-ROUGE cases, because the model would learn the specific way this corpus's articles are structured. A secondary investment would be a faithfulness classifier — a lightweight model trained to flag summaries that contain claims not grounded in the source — to catch the cases where ROUGE looks acceptable but the output quietly drifts from the article. ROUGE alone, as Section 3 shows, cannot make that distinction.

---

## Section 6: Limits of the Evaluation

Two limits stand out as directly relevant to the production scenarios discussed in Section 4.

First, **ROUGE does not measure faithfulness**. A summary can score 0.40 on ROUGE-1 while containing a claim that does not appear anywhere in the article, and the metric will not flag it. The faithfulness check in Section 3 covered three summaries by hand, which is not enough to characterize the system at scale. For the newsroom digest scenario in Section 4, this is a meaningful risk: a journalist or editor relying on auto-generated summaries needs to know not just that they read like the article, but that nothing in them is wrong. ROUGE gives no assurance on that front.

Second, **these numbers are single-request CPU latency estimates, not throughput under load**. The roughly 3 seconds per article noted in the comparison table is the time to run one inference on an unloaded machine. In a production setting — especially the real-time trading dashboard scenario, where latency budgets are tight and requests arrive concurrently — throughput, batching behavior, and tail latency under contention are what actually matter. None of those are captured here. A model that scores well on quality metrics but saturates a CPU server under moderate load is not production-ready, and this evaluation gives no signal on that question.