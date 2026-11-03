# Evaluation report: 20261102-101928-v2-final

- Split: **test** (44 questions: 30 answerable, 14 unanswerable; 0 not yet hand-validated)
- Dataset: 1.1 (labels `770ad0c4f3b5`, file `1c0a54866acd`)
- Knowledge base pin: matches
- Retriever: `hybrid_rrf_v2`, top_k=4, embedding `sentence-transformers/all-MiniLM-L6-v2`, chunks 800/120
- Generator: `openai/gpt-oss-20b`, prompt `v2`
- Git: `6877042ac9`

## Retrieval (document level, N = 30 answerable questions)

| k | Hit@k | MRR@k |
| --- | --- | --- |
| 1 | 1.000 | 1.000 |
| 3 | 1.000 | 1.000 |
| 4 | 1.000 | 1.000 |
| 5 | 1.000 | 1.000 |
| 10 | 1.000 | 1.000 |

### By question type (k = 4)

| question type | n | Hit@4 | MRR@4 |
| --- | --- | --- | --- |
| ambiguous | 4 | 1.000 | 1.000 |
| confusable | 8 | 1.000 | 1.000 |
| direct | 4 | 1.000 | 1.000 |
| error_code | 6 | 1.000 | 1.000 |
| multi_chunk | 8 | 1.000 | 1.000 |

### By category (k = 4)

| category | n | Hit@4 | MRR@4 |
| --- | --- | --- | --- |
| account_access | 3 | 1.000 | 1.000 |
| email_collaboration | 5 | 1.000 | 1.000 |
| facilities_maintenance | 5 | 1.000 | 1.000 |
| general | 2 | 1.000 | 1.000 |
| hardware_peripherals | 4 | 1.000 | 1.000 |
| network_wifi | 3 | 1.000 | 1.000 |
| printing_office_equipment | 3 | 1.000 | 1.000 |
| software_applications | 5 | 1.000 | 1.000 |

## Generation and abstention

- Attempted 44, succeeded 44, failed 0
- Abstention recall: 0.929 (13/14 unanswerable)
- False abstention rate: 0.033 (1/30 answerable)
- Abstention precision: 0.929 (13/14 abstentions)

## Latency (ms)

| stage | n | median | p95 |
| --- | --- | --- | --- |
| retrieval (eval pass) | 44 | 19.1 | 27.8 |
| retrieval (in pipeline) | 44 | 19.1 | 30.5 |
| generation | 44 | 860.5 | 2314.1 |
| total | 44 | 878.0 | 2332.6 |

## Efficiency

- Tokens: 75913 total (60021 prompt, 15892 completion); average 1725 per question
- Estimated generation cost: n/a (prices not configured)

## Review manually: answered although the knowledge base cannot answer

- `q075` Where can I download the Meridian DeskView mobile app for my personal Android phone?
  > The company policy does not allow installing a personal copy of Meridian DeskView from the internet. Instead, you should use the Meridian App Shelf or submit a request for review to obtain the app for your personal Android phone. [2]

## Review manually: abstained although the knowledge base can answer

- `q046` The installer for a catalogue app rolls back before launch; it never reaches the sign-in screen. Should I use the runtime cache-cleanup steps?
  > The excerpts do not mention “runtime cache‑cleanup steps.” They advise using the built‑in Refresh command, signing out and back in, and clearing the application’s temporary local cache only through the Settings panel if that option is available.

## RAGAS (LLM-judge scores)

- Judge `openai/gpt-oss-120b` via https://api.groq.com/openai/v1 (ragas 0.4.3); answer-relevancy embeddings `sentence-transformers/all-MiniLM-L6-v2`
- Same model family as the generator: **yes (scores may be inflated)**

| metric | scored | failed | skipped | mean | median |
| --- | --- | --- | --- | --- | --- |
| faithfulness | 29 | 0 | 15 | 0.863 | 1.000 |
| answer_relevancy | 29 | 0 | 15 | 0.597 | 0.576 |
| context_precision | 30 | 0 | 14 | 0.910 | 1.000 |
| context_recall | 30 | 0 | 14 | 0.900 | 1.000 |

Denominators: faithfulness and answer relevancy are scored on answerable questions the system answered; context precision and recall on answerable questions with a successful generation (retrieved contexts and reference answer). Skips and failures, with reasons, are in `ragas_scores.jsonl`. These are judge scores, not verified ground truth.

## Metric definitions and denominators

- **Hit@k**: share of answerable questions where at least one of the top-k retrieved chunks comes from a reference document. Denominator: answerable questions.
- **MRR@k**: mean over answerable questions of 1/rank of the first retrieved chunk from a reference document within the top k (0 if none). Ranks count chunks, not unique documents.
- **Abstention**: the pipeline flag `answerable=false` (the model emitted INSUFFICIENT_EVIDENCE, or retrieval returned nothing). Chat intent routing is not part of this evaluation.
- **Abstention recall**: correct abstentions / unanswerable questions (failed generations excluded). **False abstention rate**: abstentions on answerable questions / answerable questions. **Abstention precision**: correct abstentions / all abstentions.
- **Latency**: `retrieval (eval pass)` is a separate warm pass over all questions; generation and total come from successful LLM calls only. p95 uses the nearest-rank method. LLM latency includes network variance.
- **Tokens**: provider-reported usage of generation calls only (judge calls are not included). **Cost** is shown only when prices are configured; API responses do not include prices.
- **Failures**: a failed generation is listed below with its reason and excluded from generation metrics. Retrieval metrics do not depend on the LLM.
- Metrics from an LLM judge (RAGAS, added in the next stage) are judge scores, not independently verified ground truth.

