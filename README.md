# HelpDeskAI

An AI helpdesk assistant for IT support and facilities, built as a retrieval-augmented generation (RAG) system with ticket management and a first-class evaluation framework. Employees chat with a bot that answers from a knowledge base with cited sources, declines what the knowledge base cannot support, and raises tickets that admins manage in a web portal.

**Status:** Version 1 (a dense-retrieval baseline) is frozen at git tag `v1.0`. Version 2 (advanced retrieval) will be compared against it on the same held-out dataset.

> **The knowledge base is synthetic.** It describes a fictional company ("Meridian Works"). Nothing in it is an official company policy, and the documents are labelled as samples.

## Contents

1. [Features](#features) 2. [Architecture](#architecture) 3. [Tech stack](#tech-stack) 4. [Repository layout](#repository-layout) 5. [Setup](#setup) 6. [Configuration](#configuration) 7. [Running the app](#running-the-app) 8. [API overview](#api-overview) 9. [Knowledge base and ingestion](#knowledge-base-and-ingestion) 10. [Tickets](#tickets) 11. [Evaluation](#evaluation) 12. [V1 baseline results](#v1-baseline-results) 13. [Reproducing the baseline](#reproducing-the-baseline) 14. [Limitations](#limitations) 15. [Security notes](#security-notes) 16. [Roadmap: Version 2](#roadmap-version-2)

## Features

- **Employee chat** (login required): answers grounded in the knowledge base with cited source documents, an explicit fallback when evidence is insufficient, rejection of out-of-scope questions, and conversations that persist across sessions.
- **Tickets from chat**: the bot offers a ticket when it cannot answer, collects missing details, shows a summary, and creates the ticket only after an explicit confirmation.
- **Admin portal**: dashboard counts, ticket cards, filters and search, assignment, status changes and resolution notes, overdue highlighting, the linked chat transcript, and knowledge-base upload, delete and re-index.
- **Evaluation framework**: a versioned, hand-validated dataset; retrieval, abstention, generation, ticket and classification metrics; RAGAS scoring; reproducible run artifacts; and a run-comparison tool for V1 vs V2.

## Architecture

```
Employee UI --> /api/chat --> ChatService --+--> intent (rules, then a small LLM call)
                                            +--> ticket flow (deterministic state machine)
                                            +--> RAG pipeline --> Retriever (FAISS) --> LLM (Groq)
Admin UI -----> /api/admin, /api/tickets -------------------------> Ticket service --> SQLite
```

**A chat turn:** the user logs in and sends a message. Greetings and explicit "create a ticket" requests are handled by rules. Anything else goes to a small LLM classifier that returns `kb_question`, `ticket_request`, `chitchat` or `out_of_scope`. A knowledge question runs the RAG pipeline: embed the query, retrieve the top 4 chunks by cosine similarity, build a prompt with numbered excerpts, and call the LLM. The reply is parsed for an abstention marker and verified citations. The whole turn is stored in SQLite.

**Design decisions**

- **The backend owns the workflow; the LLM only suggests.** Intents, ticket categories and priorities are validated against fixed sets. The LLM never touches the database.
- **Machine-readable abstention.** The prompt tells the model to begin with `INSUFFICIENT_EVIDENCE` when the excerpts do not support an answer. The pipeline strips it, sets `answerable=false`, and the chat then offers a ticket.
- **Citations are verified.** The model cites `[1]`, `[2]`. The API returns every retrieved source with a `cited` flag derived from the numbers that actually appear in the answer.
- **Retriever interface.** Retrieval sits behind `retrieve(query, k)` and a registry (`register_retriever`), so V2 retrievers plug in by name without touching chat or tickets.
- **Tickets are never created silently.** Idempotency keys scoped per user and conversation prevent duplicates, and the requester comes from the login token, never from the request body.
- **Atomic turns.** A turn's messages and state are saved together, so a failed LLM call leaves no half-written conversation.
- **Evaluation is reproducible by construction.** Every run records the dataset label hash, the knowledge-base file hashes, the configuration, package versions and the git commit.

## Tech stack

Python (developed on 3.14.2 in GitHub Codespaces), FastAPI, SQLAlchemy and SQLite, FAISS (`faiss-cpu`), sentence-transformers (`all-MiniLM-L6-v2`, CPU), the `openai` client pointed at Groq's OpenAI-compatible API (generator `openai/gpt-oss-20b`), RAGAS 0.4.3 (judge `openai/gpt-oss-120b`), argon2 password hashing, JWT access tokens, plain HTML/CSS/JavaScript frontends, `uv` for dependency management, and pytest. No GPU or paid vector database is needed.

## Repository layout

```
app/
  api/          routers: health, auth, tickets, chat, admin
  core/         config, constants (categories, statuses, SLAs), security, key pool
  database/     engine and session
  models/       SQLAlchemy models: tickets, users, conversations, messages
  schemas/      Pydantic request/response models
  tickets/      ticket business rules
  services/     chat orchestration, intent, ticket flow and drafting, users, KB admin
  rag/          chunking, embeddings, vector store, retrieval, prompts, generation, pipeline, warm-up
  evaluation/   dataset schema, metrics, runner, RAGAS wiring, ticket/classification eval, comparison
  static/       employee/ and admin/ frontends
configs/experiments/   experiment configs (v1_baseline.json)
data/knowledge_base/   synthetic KB documents (pdf, docx, txt)
data/evaluation/       questions, classification examples, ticket scenarios, dataset_meta.json
scripts/               ingest, create_user, evaluate, compare_runs, validate_dataset, check_ragas, check_keys, bench_retrieval
reports/baselines/     archived V1 baseline runs and decision record
tests/
```

## Setup

Requires `uv` and Python 3.10 or newer.

```bash
git clone <repo-url> && cd HelpDeskAI
uv venv && source .venv/bin/activate

# CPU-only PyTorch first (the default build pulls multi-GB CUDA libraries)
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements.txt

cp .env.example .env   # then fill in the secrets (see Configuration)
```

Notes:

- If resolving the pinned `torch` line in `requirements.txt` fails, delete that line (torch was installed in the previous step) and re-run the second command.
- **`langchain-community` must stay below 0.4.2.** RAGAS 0.4.3 imports a module that newer `langchain-community` releases removed. The pin is recorded in `requirements.txt`; do not upgrade it without re-running `python -m scripts.check_ragas`.
- The first run downloads the embedding model (about 90 MB).

## Configuration

Copy `.env.example` to `.env`. Never commit `.env` or paste keys into chats or logs.

| Variable | Purpose | Default |
| --- | --- | --- |
| `JWT_SECRET` | Signs employee login tokens. 32+ random characters (`python -c "import secrets; print(secrets.token_hex(32))"`). | required |
| `ADMIN_API_KEY` | Development-only admin guard (see Security notes). | required |
| `GROQ_API_KEY` or `GROQ_API_KEYS` | LLM access. `GROQ_API_KEYS` takes several comma-separated keys used round-robin with failover. Groq documents its rate limits as per organization, so keys from one organization share a pool. | required |
| `GROQ_MODEL` | Generator model. | `openai/gpt-oss-20b` |
| `GROQ_REASONING_EFFORT` | Optional (`low`, `medium`, `high`). | unset |
| `LLM_MAX_TOKENS` | Completion budget (includes reasoning tokens for gpt-oss models). | `4096` |
| `EMBEDDING_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `RETRIEVAL_TOP_K`, `RETRIEVER_NAME` | Retrieval settings. | MiniLM, 800, 120, 4, `dense_v1` |
| `WARMUP_ON_STARTUP` | Load the model and index when the server starts. | `true` |
| `RAGAS_JUDGE_MODEL`, `RAGAS_EMBEDDING_MODEL`, `RAGAS_CONCURRENCY` | Evaluation judge settings. | `openai/gpt-oss-120b`, MiniLM, 4 |
| `KB_MAX_UPLOAD_MB` | Admin upload size cap. | `10` |

## Running the app

```bash
python -m scripts.ingest                                   # build the FAISS index from data/knowledge_base
python -m scripts.create_user alice --full-name "Alice Rao"   # create an employee (prompts for a password)
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- Employee chat: `/employee/` (the root redirects here)
- Admin portal: `/admin/` (sign in with `ADMIN_API_KEY`)
- Interactive API docs: `/docs`

Use `--reload` only while developing; it repeats the model warm-up on every code change. Admin uploads and index rebuilds can also be done from the admin portal's Knowledge base tab.

## API overview

| Endpoint | Auth | Purpose |
| --- | --- | --- |
| `GET /health`, `GET /ready` | none | liveness; readiness (database and index status) |
| `POST /api/auth/login`, `GET /api/auth/me` | login / employee token | sign in, current user |
| `GET /api/meta` | none | categories, priorities, statuses, allowed transitions, team members |
| `POST /api/chat`, `GET /api/chat/{id}` | employee token | send a message; restore a conversation |
| `POST /api/tickets` | employee token | create a ticket (requester taken from the token) |
| `GET/PATCH /api/tickets`, `GET /api/tickets/{id}` | admin key | list with filters, read, update |
| `GET /api/admin/stats`, `GET /api/admin/tickets/{id}/conversation` | admin key | dashboard counts; ticket transcript |
| `GET /api/admin/kb`, `POST /api/admin/kb/documents`, `DELETE /api/admin/kb/documents/{file}` | admin key | list, upload, delete knowledge-base files |
| `POST/GET /api/admin/kb/reindex` | admin key | start a background rebuild; poll status |

## Knowledge base and ingestion

- Supported files: PDF (text layer), DOCX (paragraph text; tables are not extracted), TXT and Markdown.
- Each document starts with `Title:` and `Category:` header lines. The document ID is the file name without its extension.
- Documents are split into paragraph-aware chunks (800 characters, 120 overlap) and embedded with `all-MiniLM-L6-v2`.
- The index is a FAISS `IndexIDMap(IndexFlatIP)` over normalised vectors (cosine similarity). It is saved with `chunks.jsonl` and `manifest.json` (embedding model, chunk settings and a SHA-256 per file). Loading checks that vector and chunk counts match, and the retriever refuses an index built with a different embedding model.
- Admin uploads are validated (type, size, magic bytes, a trial text extraction, and document-ID collisions) before they can reach the index.
- The included knowledge base has 15 synthetic documents (PDF, DOCX and TXT) across seven helpdesk categories plus a general FAQ.

## Tickets

- **Categories:** network and Wi-Fi, hardware and peripherals, software and applications, email and collaboration, account access, printing and office equipment, facilities and maintenance.
- **Lifecycle:** `open -> in_progress -> resolved -> closed`, with `in_progress -> open` and `resolved -> in_progress` allowed; closed is terminal. Resolving requires resolution notes. The backend validates every transition.
- **Priorities and due dates:** low, medium, high, critical. A due date is set from the priority (72, 24, 8 and 4 hours). A ticket is overdue when it is past due and still open or in progress.
- Each ticket stores a transcript of the linked conversation for the admin view. The sample helpdesk team members in `app/core/constants.py` are fictional.

## Evaluation

The evaluation framework is a first-class part of the project. It works independently of the web UI.

**Dataset (`data/evaluation/qa_v1.jsonl`).** 42 questions: 21 dev and 21 test. Types: 8 direct, 5 multi-chunk, 6 paraphrase, 4 ambiguous, 6 error-code, 5 confusable and 8 unanswerable. Each answerable question stores a reference answer, the reference document IDs, and verbatim quotes from those documents (a validator checks that the quotes really exist). Questions were drafted with an LLM from the knowledge base and then validated by hand against the documents; each row records `origin` and `validated`. Paraphrase groups never span splits. `dataset_meta.json` pins the SHA-256 of every knowledge-base file, and runs refuse to start if the indexed KB differs. Dev is for debugging; **test is held out** and was run for the final V1 report.

**Metrics and denominators**

- **Retrieval, document level, answerable questions only:** Hit@k and MRR@k at k = 1, 3, 4, 5 and 10. No LLM is involved, so these are deterministic.
- **Abstention** (pipeline flag): recall on unanswerable questions, false-abstention rate on answerable questions, and precision.
- **RAGAS (LLM judge):** faithfulness and answer relevancy on answerable questions the system answered; context precision and context recall on answerable questions with a successful generation. Unanswerable questions are covered by the abstention metrics instead. Every skip and failure is recorded with its reason.
- **Tickets:** creation success rate, required-field completeness and end-to-end category agreement, using the real chat flow against a throwaway database.
- **Category classification:** accuracy and per-class and macro precision, recall and F1. LLM errors are excluded and counted; an unclassified prediction counts as wrong.
- **Latency** (median and p95, retrieval separate from generation), **token usage**, and **cost** (only when prices are configured; API responses do not include prices).
- Failed generations are listed with their reason and excluded from generation metrics; retrieval metrics do not depend on the LLM.

**Commands**

```bash
python -m scripts.validate_dataset                       # schema, quotes vs KB, coverage, KB pin
python -m scripts.evaluate --split dev --retrieval-only  # deterministic retrieval metrics, no API calls
python -m scripts.evaluate --split smoke --ragas         # quick end-to-end check
python -m scripts.evaluate --split test --name my-run --all --sleep 1
python -m scripts.evaluate --rescore <run_id> --ragas    # re-score saved generations only
python -m scripts.compare_runs <baseline_run> <candidate_run>
```

Each run writes `config.json`, `retrieval.jsonl`, `generations.jsonl`, `per_question.csv`, `ragas_scores.jsonl`, `tickets.jsonl`, `classification.jsonl`, `summary.json` and a readable `report.md` to `reports/runs/<run_id>/`. `compare_runs` refuses runs with different labels, KB versions or judges (unless forced) and reports paired bootstrap 95% intervals.

## V1 baseline results

Run `20261027-123340-v1-baseline-r2`: test split (21 questions: 16 answerable, 5 unanswerable), commit `a78d1816f1` with a clean working tree, dense retrieval with top_k = 4, prompt `v1`, generator `openai/gpt-oss-20b` (`LLM_MAX_TOKENS=4096`), RAGAS judge `openai/gpt-oss-120b` (the same model family as the generator, which can inflate judge scores). Archived in `reports/baselines/` with the decision record in `reports/baselines/DECISIONS.md`.

**Retrieval** (N = 16 answerable)

| k | Hit@k | MRR@k |
| --- | --- | --- |
| 1 | 0.875 | 0.875 |
| 3 | 0.938 | 0.896 |
| 4 | 1.000 | 0.911 |
| 5 and 10 | 1.000 | 0.911 |

By type at k = 4, every type has Hit@4 = 1.000. MRR@4 is 1.000 for direct, error-code, multi-chunk, ambiguous and paraphrase questions, and **0.528 for confusable questions (n = 3)**. All the ranking loss sits in that type: the right document was retrieved but ranked third or fourth.

**Generation and abstention**

| metric | value |
| --- | --- |
| Generations succeeded | 21 / 21 |
| Abstention recall | 1.000 (5 / 5 unanswerable) |
| False-abstention rate | 0.062 (1 / 16 answerable) |
| Abstention precision | 0.833 (5 / 6 abstentions) |

**RAGAS** (LLM-judge scores, not ground truth)

| metric | scored | mean | median |
| --- | --- | --- | --- |
| faithfulness | 15 | 0.837 | 0.850 |
| answer relevancy | 15 | 0.558 | 0.552 |
| context precision | 16 | 0.927 | 1.000 |
| context recall | 16 | 0.969 | 1.000 |

**Tickets and classification**

| metric | value | n |
| --- | --- | --- |
| Ticket creation success, field completeness, category agreement | 1.000 each | 5 scenarios |
| Classification accuracy, macro F1 | 0.929, 0.924 | 14 examples |

The single classification error was `email_collaboration -> software_applications`.

**Latency and cost** (GitHub Codespaces, CPU)

| stage | median | p95 |
| --- | --- | --- |
| retrieval, inside the pipeline | 15.7 ms | 25.9 ms |
| LLM generation | 843 ms | 1,642 ms |
| total reply | 863 ms | 1,668 ms |

Tokens: 32,821 total, about 1,563 per question. Cost was not estimated because prices were not configured. A separate warm retrieval pass over the same questions measured a higher median of 41.2 ms (p95 171.5 ms) in this run and a similar value in the earlier run; the cause was not determined.

**How to read these numbers**

- **The sample is small.** With 16 answerable questions, one question is 6 percentage points. Hit@4 of 16/16 is consistent with a true rate anywhere from roughly 79% to 100% (95% interval), and abstention recall of 5/5 with roughly 48% to 100%. Per-class classification results rest on 2 examples each.
- **Hit@4 is saturated.** Version 2 cannot improve it. Differences can only show up in ranking (Hit@1, MRR), in the RAGAS context metrics, and in answer quality.
- **Document-level labels hide chunk-level misses.** Question q013 asks two things (what to put in a ticket and the acknowledgement time for high priority). The right document was retrieved, yet the model reported insufficient evidence for part of it and abstained. It was a false abstention in both V1 runs, which suggests a systematic gap: either the needed chunk was not retrieved or the prompt handles partial answers poorly.
- **RAGAS faithfulness is not a hallucination rate.** The V1 prompt lets the model add a clearly labelled "General suggestion (not from company documentation)" paragraph. By design those sentences are not supported by the excerpts, and the judge may count them against faithfulness. Answer relevancy is only meaningful relative to another run using the same embedding model.
- **LLM-judge scores are noisy.** An earlier V1 run (`20261025-164815-v1-baseline`, r1) gave faithfulness 0.744, answer relevancy 0.610, context precision 0.917 and context recall 0.967 on the same questions. The runs are not pure repeats: r1 lost one generation (q009) to token-budget truncation, and generation is non-deterministic. Differences between runs smaller than about 0.05 to 0.1 should not be read as real. Retrieval metrics were identical in both runs.
- **Run history.** r1 had one generation fail (`finish_reason=length` at a 2,048-token budget). The decision to re-run once with a 4,096-token budget, and to treat that run (r2) as the baseline whatever its numbers, was recorded in `reports/baselines/DECISIONS.md` before r2 ran. Both runs are kept.

## Reproducing the baseline

1. Check out tag `v1.0`, install as in [Setup](#setup), and fill `.env`. Retrieval-only runs need no API key.
2. Build the index from the committed knowledge base: `python -m scripts.ingest`.
3. Confirm the KB pin: `python -m scripts.validate_dataset` must report that it matches.
4. Retrieval metrics (deterministic, no API calls): `python -m scripts.evaluate --split test --retrieval-only`. These should match the table above exactly.
5. Full run: `python -m scripts.evaluate --split test --name v1-repro --all --sleep 1`. Generation, RAGAS, ticket and classification metrics depend on LLM sampling, so expect small differences. Compare against `reports/baselines/` with `python -m scripts.compare_runs`.

Requirements for a fair comparison: the same dataset labels, knowledge-base files, `LLM_MAX_TOKENS` and judge. `compare_runs` checks these and refuses otherwise.

## Limitations

**Evaluation**
- Small dataset: 21 test questions, 5 unanswerable and 5 ticket scenarios. Results are indicative, not conclusive.
- All 8 unanswerable questions concern HR, payroll or finance, which are easy to decline. There are no IT-sounding questions that the knowledge base merely does not cover, which is the harder abstention case, so the abstention numbers probably flatter V1.
- Some confusable questions list a single reference document although a second document could reasonably answer them, which can lower MRR. Two questions that share a group (q001 and q014) test different facts.
- Questions are single-document and written in clean English; there are no typo-heavy, long or Hinglish queries.
- The RAGAS judge shares a model family with the generator. The generator and judge ran on one provider, and the knowledge base is synthetic, written by an LLM.
- **No adversarial testing was run.** Prompt injection, abusive or malformed input, secret and system-prompt extraction, and ticket manipulation were not evaluated. The V1 defences (delimited excerpts, an instruction to ignore instructions inside them, and backend validation of every LLM suggestion) are design-level and untested.

**Product**
- Retrieval is dense-only, with no query rewriting, so follow-up questions are retrieved without conversation context.
- Category inference in the chat flow is keyword-assisted and the employee confirms it; it is not a trained classifier.
- No login rate limiting or lockout, no pagination on the ticket list, and a re-index runs inside the API process (it can slow queries while it runs).
- Latency is dominated by the hosted LLM; whole-reply latency cannot approach 100 ms with a remote LLM in the loop.

## Security notes

- Secrets live in environment variables; `.env` is not committed, keys are not logged (logs use labels such as `key-2/5`), and the frontend never sees them.
- Passwords are hashed with argon2; login responses are identical for unknown users and wrong passwords.
- The frontends render all text with `textContent` (no `innerHTML`).
- **The admin portal uses a shared development key.** Production needs real authentication, authorization and secure session handling.
- Employee tokens are kept in `sessionStorage`; production should use httpOnly, secure cookies.
- Administrators can read the chat transcripts linked to tickets.
- Deploy behind TLS, add login rate limiting, and use a secrets manager.

## Roadmap: Version 2

V2 changes retrieval and prompting while keeping the ticket system, the dataset and the metric definitions fixed:

- BM25 and hybrid dense + sparse retrieval with reciprocal rank fusion.
- Cross-encoder reranking, query rewriting, better chunking and metadata filters.
- Prompt optimisation for partial answers, abstention and grounding (the q013 behaviour and the faithfulness interaction above).
- Before V2 runs: more questions of the hard types (confusable, multi-chunk, error-code, IT-adjacent unanswerable), written without looking at either system's outputs, plus an adversarial test set.

Add `configs/experiments/v2_*.json`, register the new retriever, run the same split, and compare with `python -m scripts.compare_runs`. The expected measurable effects are on Hit@1, MRR, the RAGAS context metrics and faithfulness; Hit@4 is already saturated.