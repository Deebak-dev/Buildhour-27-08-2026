# PRD — Mutual Fund FAQ Assistant (Facts-Only RAG Chatbot)

| Field | Value |
| --- | --- |
| Document | Product Requirements Document (PRD) |
| Version | 1.0 (draft for review) |
| Date | 2026-09-27 |
| Project type | Class milestone — working prototype |
| Source brief | `docs/Problemstatement.txt` |
| Status | Awaiting approval before implementation |

---

## 1. Overview

Build a small **facts-only FAQ assistant** for **HDFC Mutual Fund schemes**. Users ask plain questions about
expense ratio, exit load, minimum SIP, ELSS lock-in, riskometer, benchmark, and how to download statements.
The assistant answers using **only official public pages**, always with **exactly one source link**, and
**never gives investment advice**.

The product is a **RAG (Retrieval-Augmented Generation) chatbot**: the answer is always grounded in
retrieved source chunks, never in the model's memory. Every stage of the RAG pipeline
(Loading → Chunking → Embedding → Vector Store → Retrieval → Generation) is a distinct, reviewable stage.

### 1.1 The problem

Retail users comparing mutual fund schemes read the same handful of facts repeatedly across many
scattered pages. Support and content teams answer the same questions repeatedly by hand. Search engines
surface blog content with opinions and outdated numbers mixed in.

The assistant reduces this to **one trusted, cited, short answer** — and refuses anything that is not a
verifiable fact about a specific scheme.

### 1.2 Product principle

> **Facts only. No advice. One source, always.**

If the answer is not a fact retrievable from the corpus, the assistant does not answer it — it declines
and points to an educational resource.

---

## 2. Goals

| # | Goal | Success signal |
| --- | --- | --- |
| G1 | Answer factual scheme questions accurately from a closed corpus | ≥ 90% of the sample Q&A set answered correctly, each with a valid citation |
| G2 | Guarantee every answer is traceable | 100% of answers contain exactly one working source link |
| G3 | Never give advice, opinions, or performance claims | 100% refusal on the adversarial/advice test set |
| G4 | Never accept or store PII | 100% block rate on the PII test set; zero PII persisted to disk or logs |
| G5 | Stay short and readable | Every answer ≤ 3 sentences |
| G6 | Show a working end-to-end RAG pipeline | Pipeline stages are visible/inspectable in code and in the README |

### 2.1 Non-goals (explicitly out of scope)

- Portfolio advice, fund recommendations, buy/sell/hold guidance, asset allocation, tax planning.
- Computing, comparing, ranking, or projecting **returns, CAGR, NAV trends, or performance**.
- SIP calculators, goal planning, portfolio trackers, watchlists, account login.
- Fund comparison tables across schemes.
- Live NAV, real-time prices, or data older than the ingested snapshot.
- Multi-user accounts, authentication, production deployment, mobile app.
- Hindi/regional language support (English only in v1).

---

## 3. Users

| User | Need | Success looks like |
| --- | --- | --- |
| **Retail investor (primary)** | "What is the exit load on HDFC Large Cap?" | Gets one short sentence + a link to the official page in under 10 seconds |
| **Retail investor (comparing)** | "What is the minimum SIP for the ELSS tax saver fund?" | Correct fact, correctly attributed to the right scheme, no steer toward buying |
| **Support / content team (secondary)** | Answer repetitive MF questions consistently | Same question always produces the same grounded answer, with a citation they can paste |

---

## 4. Scope

### 4.1 In scope — AMC and schemes

**AMC: HDFC Mutual Fund (HDFC AMC).** Five schemes, all **Direct – Growth** variants:

| # | Scheme (as cited) | Category | Seed URL |
| --- | --- | --- | --- |
| 1 | HDFC Large Cap Fund – Direct Growth | Large Cap | `https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth` |
| 2 | HDFC Equity Fund – Direct Growth | Flexi Cap | `https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth` |
| 3 | HDFC ELSS Tax Saver Fund – Direct Growth | ELSS (80C) | `https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth` |
| 4 | HDFC Small Cap Fund – Direct Growth | Small Cap | `https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth` |
| 5 | HDFC Balanced Advantage Fund – Direct Growth | Hybrid | `https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth` |

Each scheme is a **separate retrieval scope**. Answers must never blend facts across schemes.

### 4.2 In scope — corpus

Public pages only, from **AMC / SEBI / AMFI** primary sources and the provided scheme pages:

- Scheme **factsheets** (current quarter)
- **KIM** (Key Information Memorandum) and **SID** (Scheme Information Document)
- **Scheme FAQ** pages
- **Fee / charge / expense** pages
- **Riskometer** and **benchmark** notes
- **Statement / tax-document** download guides (capital-gains, tax reports)

### 4.3 Out of scope as sources

Third-party blogs, news articles, social media, YouTube, forums, Wikipedia, aggregator blogs,
AI-generated summaries, and any site that offers advice. *(Note: the seed URLs are on Groww, a
registered distributor platform — see §12 Risk R1.)*

### 4.4 Question types the assistant must handle

| Type | Example query |
| --- | --- |
| Fees | "What is the expense ratio of HDFC Large Cap Fund – Direct Growth?" |
| Exit load | "Is there an exit load on HDFC Equity Fund?" |
| Minimum SIP | "What is the minimum SIP amount for HDFC ELSS Tax Saver Fund?" |
| Lock-in | "What is the lock-in period in HDFC ELSS Tax Saver Fund?" |
| Tax benefit | "What is the section 80C limit for HDFC ELSS Tax Saver Fund?" |
| Riskometer | "What is the current riskometer level of HDFC Small Cap Fund?" |
| Benchmark | "What is the benchmark of HDFC Balanced Advantage Fund?" |
| Documents | "How do I download my capital gains statement?" |
| Orientation | "What is a direct plan vs a regular plan?" |
| Scheme identity | "Who manages HDFC Large Cap Fund and what is the benchmark index?" |

---

## 5. Constraints and guardrails (hard requirements)

### 5.1 Facts-only, no advice

- **Must refuse** opinionated, predictive, or portfolio questions with a polite, facts-only message plus
  one relevant educational link. Examples: "Should I buy HDFC Small Cap Fund?", "Which fund is best
  for me?", "Is now a good time to invest?", "Is 20% in equity too much for me?"
- **Must not** state or imply a recommendation, a "best" scheme, a suitability judgement, or a
  preference between schemes.
- **Must not** compare schemes against each other in a way that ranks them.

### 5.2 No performance claims

- Must **never** compute, quote, compare, or extrapolate returns, CAGR, NAV movement, or rankings.
- If asked about performance ("How has HDFC Large Cap Fund performed?"), respond by **linking to the
  official factsheet** and stating that performance figures are not provided by this assistant.

### 5.3 No PII

- Must **not accept, echo, or store**: PAN, Aadhaar, account numbers, OTP, email addresses, phone numbers,
  or any Demat/Folio number.
- Input is screened **before** retrieval. PII-bearing input is rejected with a neutral message and the
  text is **never persisted** to chat history, vector store, or logs.
- No login, no account linkage, no third-party PII transmission. The app uses no user account at all.

### 5.4 Public sources only

- Corpus contains only public pages. No paywalled, login-gated, or personal-report content.
- The source list deliverable must be complete and reviewable.

### 5.5 Clarity and transparency

- Every answer is **≤ 3 sentences**.
- Every answer ends with a source line and a freshness line.
- No app back-end screenshots in any deliverable.
- If the corpus does not contain the answer, say so — never guess, never fall back to model knowledge.

---

## 6. Functional requirements

### FR-1 — Corpus ingestion (Loading)

| ID | Requirement |
| --- | --- |
| FR-1.1 | Fetch each of the 5 seed scheme pages and the primary AMC/SEBI/AMFI documents listed in §4.2. |
| FR-1.2 | Extract main readable text; strip navigation, footer, cookie banners, ads, and script content. |
| FR-1.3 | Attach metadata to every document: `scheme`, `category`, `doc_type`, `source_url`, `publisher`, `ingested_at`, `doc_title`. |
| FR-1.4 | Persist both the **raw snapshot** and the **cleaned text** per document, so the crawl is reproducible and auditable. |
| FR-1.5 | Log and record failures per URL; a failed fetch must not silently produce an empty document. |
| FR-1.6 | Record the retrieval/crawl timestamp — this drives the "Last updated from sources" line. |

### FR-2 — Chunking

| ID | Requirement |
| --- | --- |
| FR-2.1 | Chunking strategy is **decided from the observed data**, not hardcoded up front: inspect real page/factsheet structure, then choose and record the parameters. |
| FR-2.2 | Preferred default (to be validated against the actual data): structure-aware chunking, target ~400–800 tokens, ~10–15% overlap. |
| FR-2.3 | Keep atomic units intact: one FAQ Q→A pair per chunk; one logical fact per chunk; a fee/charge table row kept with its column headers. |
| FR-2.4 | Every chunk inherits and may extend its document metadata with `chunk_id`, `section`, and `field_type` (e.g. `expense_ratio`, `exit_load`, `min_sip`, `lock_in`, `riskometer`, `benchmark`). |
| FR-2.5 | The chosen strategy, the reason, and the final parameters must be documented in the README. |

### FR-3 — Embedding

| ID | Requirement |
| --- | --- |
| FR-3.1 | Use `sentence-transformers/all-MiniLM-L6-v2` as the embedding model. |
| FR-3.2 | Embeddings are 384-dimensional; this is recorded in the README. |
| FR-3.3 | The embedding model name and dimension are stored in the collection metadata. |
| FR-3.4 | Embedding is run offline/local — no paid embedding API required. |

### FR-4 — Vector store

| ID | Requirement |
| --- | --- |
| FR-4.1 | Store vectors and metadata in **ChromaDB** (persistent local client). |
| FR-4.2 | The collection name and dimension are explicit, and re-ingestion is idempotent (re-running rebuilds rather than duplicates). |
| FR-4.3 | Metadata filtering by `scheme` and `field_type` must be supported at query time. |

### FR-5 — Retrieval

| ID | Requirement |
| --- | --- |
| FR-5.1 | Retrieve top-k semantically similar chunks (default k = 5), with a similarity-score threshold below which the assistant treats the corpus as having no answer. |
| FR-5.2 | Apply a `scheme` metadata filter when the query identifies a scheme, to prevent cross-scheme contamination. |
| FR-5.3 | Detect and prioritize chunks matching the query's `field_type` (e.g. an "expense ratio" query prefers chunks tagged `expense_ratio`). |
| FR-5.4 | Optional, if time allows: keyword/BM25 hybrid search merged with vector search, and a cross-encoder re-ranker. Both are explicitly labelled as stretch goals. |

### FR-6 — Answer generation

| ID | Requirement |
| --- | --- |
| FR-6.1 | The LLM receives **only** the retrieved chunks as context. It is instructed to answer from context only and to refuse if the context is insufficient. |
| FR-6.2 | Answers are **≤ 3 sentences**, factual, plain English, no marketing tone, no second person advice ("you should"). |
| FR-6.3 | Every answer includes **exactly one** source link, taken from the metadata of the chunk it was grounded in. |
| FR-6.4 | Every answer includes the freshness line `Last updated from sources: <date>`. |
| FR-6.5 | If the retrieved context is insufficient or the similarity threshold is not met, return the "not in my sources" response — never generate from model knowledge. |
| FR-6.6 | Prompt and response contract are fixed in one config/prompt module and documented in the README. |

### FR-7 — Guardrails and refusal routing

| ID | Requirement |
| --- | --- |
| FR-7.1 | A classifier/routing step runs **before** retrieval and assigns the query to one of: `FACTUAL`, `ADVICE` (refuse), `PERFORMANCE` (redirect to factsheet), `PII` (reject), `OUT_OF_SCOPE` (refuse politely). |
| FR-7.2 | Advice queries return a polite, facts-only message + one educational link. |
| FR-7.3 | Performance queries return a message + the official factsheet link, with no figures. |
| FR-7.4 | PII queries are rejected before retrieval and the input is not stored. |
| FR-7.5 | Out-of-corpus queries (e.g. "What is the NAV of HDFC Large Cap today?") return a not-in-sources response. |

### FR-8 — User interface

| ID | Requirement |
| --- | --- |
| FR-8.1 | **Welcome line** on load. |
| FR-8.2 | **Exactly 3 example questions** shown as clickable suggestions. |
| FR-8.3 | A permanent visible note: **"Facts-only. No investment advice."** |
| FR-8.4 | Every answer renders: answer text, one citation link, and the freshness line. |
| FR-8.5 | The source link is visible in the answer, not hidden behind a hover or a "Sources" drawer. |
| FR-8.6 | A small, optional "Show retrieved chunks" expander for demonstrating the RAG internals in the demo. |
| FR-8.7 | No back-end/admin screens and no back-end screenshots in deliverables. |

### FR-9 — Transparency metadata

| ID | Requirement |
| --- | --- |
| FR-9.1 | App states the corpus scope: 1 AMC, 5 schemes, N documents, ingested on `<date>`. |
| FR-9.2 | The disclaimer snippet is shown in the UI and stored as a single source-of-truth constant. |

---

## 7. Response contract

Every factual answer follows this exact shape:

```
<Answer, maximum 3 sentences, facts only.>

Source: <one URL>
Last updated from sources: <DD MMM YYYY>
```

Required refusal shape (advice / performance / out-of-scope):

```
<Polite facts-only message.>
Source: <one relevant educational or official link>
Last updated from sources: <DD MMM YYYY>
```

Required PII rejection shape:

```
<Neutral message: this assistant does not accept personal or account information.>
Facts-only. No investment advice.
```

### 7.1 Worked examples

**Q:** *Expense ratio of HDFC Large Cap Fund – Direct Growth?*
**A:** The expense ratio (annual TER) for HDFC Large Cap Fund – Direct Growth is **X.XX%** as stated on the
source page. This is the annual ongoing charge deducted from the scheme's assets; it is not a one-time fee.
Direct plans generally have a lower expense ratio than regular plans.
`Source: <one link>` · `Last updated from sources: <date>`

**Q:** *Should I buy HDFC Small Cap Fund?*
**A:** I'm a facts-only assistant and don't give buy, sell, or hold guidance, and I don't compare or rank
schemes. For suitability and suitability-based allocation, please read a SEBI-registered investment
adviser's guidance and the scheme's SID.
`Source: <SEBI / educational link>` · `Last updated from sources: <date>`

**Q:** *How has HDFC Large Cap Fund performed?*
**A:** I don't provide returns, CAGR, or performance comparisons. The official factsheet publishes the
scheme's past performance and is the right place to read it.
`Source: <official factsheet link>` · `Last updated from sources: <date>`

**Q:** *My PAN is ABCDE1234F, tell me my exit load.*
**A:** (rejected before retrieval; nothing stored)

---

## 8. Non-functional requirements

| ID | Category | Requirement |
| --- | --- | --- |
| NFR-1 | Grounding | Zero ungrounded answers. If not in the retrieved context, the assistant says so. |
| NFR-2 | Citation integrity | Every citation URL is one that exists in the ingested source list and resolves. |
| NFR-3 | Latency | End-to-end answer in ≤ 10 s on the demo machine (local embeddings, local vector DB). |
| NFR-4 | Privacy | No telemetry, no PII in logs, no external calls other than the model provider and the initial crawl. |
| NFR-5 | Reproducibility | One command re-ingests, re-embeds, and re-runs the app from a clean checkout. |
| NFR-6 | Portability | Runs on Windows (dev machine) and Linux. CPU-only is acceptable. |
| NFR-7 | Maintainability | Pipeline stages are separate modules with clear interfaces. |
| NFR-8 | Transparency | README documents the chunking decision, embedding model, collection, and limits. |
| NFR-9 | Offline embedding | Embeddings and vector search require no network. |

---

## 9. RAG pipeline (stage-by-stage)

The pipeline is the demo. Each stage is implemented, inspectable, and separately runnable.

```
        ┌──────────┐   ┌──────────┐   ┌───────────┐   ┌──────────────┐
        │ LOADING  │──▶│ CHUNKING │──▶│ EMBEDDING │──▶│ VECTOR STORE │
        │  crawl + │   │structure-│   │ MiniLM-L6 │   │   ChromaDB   │
        │  clean   │   │  aware   │   │  384-d    │   │  persistent  │
        └──────────┘   └──────────┘   └───────────┘   └──────┬───────┘
                                                            │
        ┌──────────┐   ┌──────────┐   ┌───────────┐          │
        │    UI    │◀──│  ANSWER  │◀──│ RETRIEVAL │◀─────────┘
        │ 3 ex. Qs │   │ contract │   │ top-k +   │
        │ + notice │   │ ≤3 sent. │   │  filters  │
        └──────────┘   └──────────┘   └───────────┘
                ▲
                │ pre-retrieval routing
        ┌───────┴────────────────────────────────────────┐
        │ GUARDRAILS: ADVICE · PERFORMANCE · PII · OOS  │
        └────────────────────────────────────────────────┘
```

| Stage | Input → Output | Module |
| --- | --- | --- |
| **Loading** | URLs → cleaned text + metadata + raw snapshot | `ingest/load.py` |
| **Chunking** | cleaned text → structured chunks + metadata | `ingest/chunk.py` |
| **Embedding** | chunks → 384-d vectors | `ingest/embed.py` |
| **Vector store** | vectors + metadata → Chroma collection | `ingest/store.py` |
| **Guardrails** | user query → route label (+ rejection) | `rag/guardrails.py` |
| **Retrieval** | query → filtered top-k chunks | `rag/retrieve.py` |
| **Answer generation** | chunks + prompt → contract-compliant answer | `rag/answer.py` |
| **UI** | answer → rendered chat view | `app.py` |

### 9.1 Chunking decision (to be finalised from the data)

The brief requires the strategy to be **decided by inspecting the real corpus**, not assumed. Working
baseline, subject to validation:

| Data shape observed | Strategy |
| --- | --- |
| FAQ Q→A blocks | One Q→A pair per chunk (never split a pair) |
| Factsheet fee/holdings tables | Row + its column headers as one chunk; chunk header row repeated |
| Narrative KIM/SID sections | Heading-anchored recursive split, 400–800 tokens, 10–15% overlap |
| Long unstructured prose | Recursive character split at sentence/paragraph boundaries |

**Deliberately not** a naive fixed-size character split: it breaks the Q→A and table-row units that the
fact lookups depend on.

---

## 10. Tech stack (per brief)

| Layer | Choice | Note |
| --- | --- | --- |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` | 384-d, runs locally/CPU |
| Vector DB | **ChromaDB** | Persistent local client |
| LLM | Provider-agnostic, pluggable | Must be swappable via one config value |
| UI | Lightweight chat UI | Single `app.py`; chat interface |
| Orchestration | Plain Python, stage modules | No agent framework needed |
| Storage of raw/clean docs | Local files | Kept for auditability |

> Open item: the specific LLM provider is not fixed by the brief. It is selected for a free tier and
> configured through a single env var so the pipeline is not tied to one vendor.

---

## 11. Deliverables

| # | Deliverable | Format | Status |
| --- | --- | --- | --- |
| D1 | Working prototype | App or notebook link; ≤ 3-min demo video if hosting is not possible | ☐ |
| D2 | Source list | CSV or MD — all URLs used, with publisher, doc type, and ingest date | ☐ |
| D3 | README | Setup steps, scope (AMC + 5 schemes), chunking decision, architecture, known limits | ☐ |
| D4 | Sample Q&A file | 5–10 queries with the assistant's actual answers + links | ☐ |
| D5 | Disclaimer snippet | The exact text used in the UI | ☐ |
| D6 | PRD | This document | ☑ |

### 11.1 D2 — source list schema

`source_id, scheme, doc_type, title, publisher, url, retrieved_at, sha256, status`

### 11.2 D4 — sample Q&A coverage (minimum)

| # | Query | Type |
| --- | --- | --- |
| 1 | Expense ratio of HDFC Large Cap Fund – Direct Growth | Factual |
| 2 | Exit load on HDFC Equity Fund – Direct Growth | Factual |
| 3 | Minimum SIP for HDFC ELSS Tax Saver Fund – Direct Growth | Factual |
| 4 | Lock-in period for HDFC ELSS Tax Saver Fund | Factual |
| 5 | Benchmark of HDFC Balanced Advantage Fund – Direct Growth | Factual |
| 6 | Current riskometer level of HDFC Small Cap Fund | Factual |
| 7 | How do I download my capital gains statement? | Factual (process) |
| 8 | Should I buy HDFC Small Cap Fund? | **Refusal — advice** |
| 9 | Which of these five funds has the best returns? | **Refusal — performance** |
| 10 | My PAN is XXXXX0000X and folio number is 12345678 — what is my exit load? | **Rejection — PII** |

### 11.3 D5 — disclaimer snippet (single source of truth)

```
Facts-only. No investment advice.
This assistant answers questions about publicly available mutual fund scheme
information and always links its source. It does not recommend, compare, or rate
schemes, and it does not provide returns, projections, or tax advice. Mutual fund
investments are subject to market risks; read the scheme's SID/KIM and consult a
SEBI-registered investment adviser before investing.
```

---

## 12. Risks and known limits

| ID | Risk | Impact | Mitigation |
| --- | --- | --- | --- |
| R1 | **Source authority mismatch.** The brief mandates AMC/SEBI/AMFI as sources, but the seed URLs are on Groww (a registered distributor platform). | Citations may be challenged as not-primary. | Crawl the seed pages as instructed **and** ingest the primary AMC/SEBI documents (SID, KIM, factsheet); prefer and cite the primary document wherever the same fact appears. Flagged for mentor confirmation (§13, Q1). |
| R2 | **Dynamic / JS-rendered pages** may not yield text via a simple fetch. | Empty or thin documents → weak retrieval. | Verify extraction per URL; fall back to the AMC-hosted PDF/doc for any page that yields too little text. Record failures in D2. |
| R3 | **Corpus is narrow.** Facts not on the 5 seed pages (e.g. current NAV, tax harvesting) will be unanswerable. | "Not in my sources" on valid questions. | Accepted by design; the scope note in the UI and README makes this explicit. Adding documents is a config change. |
| R4 | **MiniLM-L6-v2 is a small model.** It may blur close distinctions like "regular vs direct" or "exit load tiers". | Wrong-fact risk — the highest-severity risk here. | Strict `scheme` metadata filtering, `field_type` boosting, higher k, and a similarity floor; verify every answer in D4 against the source. |
| R5 | **Stale facts.** Expense ratios and exit loads change; riskometer levels move. | Wrong answers over time. | Freshness line is mandatory; ingest date in UI and D2; README states the snapshot is a point-in-time corpus. |
| R6 | **Guardrail bypass.** A user may phrase advice as a fact ("If someone has a 10-year horizon, which…"). | Advice leak. | Route first, enforce with the system prompt, and test the adversarial set in §14. |
| R7 | **LLM drift into numbers.** The model may round, reformat, or "correct" a figure. | Inaccurate facts. | Instruct verbatim figure reproduction; only ever pass the retrieved chunk; never let the model compute. |
| R8 | **Rate limits / crawl failure** on source pages. | Missing documents. | Cache raw snapshots locally; ingest is re-runnable and idempotent. |

---

## 13. Open questions

| # | Question | Owner | Needed by |
| --- | --- | --- | --- |
| Q1 | Are Groww pages acceptable citations, or must every citation resolve to HDFC AMC / SEBI / AMFI? (See R1.) | Mentor | Before ingestion |
| Q2 | Which LLM provider/free tier should be wired in? | Team | Before build |
| Q3 | Hosting: is a public link expected, or is the ≤ 3-min video acceptable? | Mentor | Before build |
| Q4 | Any mandatory response language, tone, or formatting beyond §7? | Mentor | Before build |
| Q5 | Is keyword/BM25 hybrid + re-ranking in scope, or is pure vector retrieval sufficient? | Mentor | After MVP |

---

## 14. Acceptance criteria

The prototype is accepted when all of the following hold.

**Functional**

- [ ] A1. All 5 schemes are in the corpus; each is individually answerable.
- [ ] A2. Each of the 10 D4 sample queries produces the expected behaviour (correct fact with link, or the specified refusal).
- [ ] A3. 100% of factual answers contain exactly one working source link.
- [ ] A4. 100% of factual answers are ≤ 3 sentences.
- [ ] A5. 100% of factual answers include `Last updated from sources: <date>`.
- [ ] A6. All advice queries are refused with a polite message + an educational link.
- [ ] A7. All performance queries are refused with a factsheet link and no figures.
- [ ] A8. All PII queries are rejected pre-retrieval, and no PII appears in any log or store.
- [ ] A9. An out-of-corpus question returns "not in my sources" rather than an invented answer.

**Pipeline**

- [ ] A10. Loading, Chunking, Embedding, Vector store, Retrieval, and Generation are separate, individually runnable stages.
- [ ] A11. Embeddings come from `sentence-transformers/all-MiniLM-L6-v2` (384-d).
- [ ] A12. Vectors are stored in ChromaDB with a persistent local collection.
- [ ] A13. The chunking strategy and its rationale are documented, justified by the observed data.
- [ ] A14. Cross-scheme contamination is checked explicitly: an HDFC Large Cap query never returns HDFC Small Cap facts.

**UI**

- [ ] A15. Welcome line is present.
- [ ] A16. Exactly 3 example questions are shown.
- [ ] A17. "Facts-only. No investment advice." is visible in the UI.
- [ ] A18. No back-end screenshots anywhere in the deliverables.

**Deliverables**

- [ ] A19. D1 prototype link or ≤ 3-min video.
- [ ] A20. D2 source list with all URLs.
- [ ] A21. D3 README with setup, scope, chunking rationale, and known limits.
- [ ] A22. D4 sample Q&A of 5–10 queries with answers and links.
- [ ] A23. D5 disclaimer snippet matching the UI text exactly.

---

## 15. Milestones

| Phase | Work | Exit criteria |
| --- | --- | --- |
| **M0 — Scope lock** | Confirm open questions (Q1–Q4), finalise corpus list, write D2 skeleton | Corpus and source policy agreed |
| **M1 — Loading** | Crawl + clean 5 scheme pages and primary AMC/SEBI/AMFI docs; raw snapshot saved | Every D2 row has content or a logged failure |
| **M2 — Chunking** | Inspect real data, decide and document strategy, produce chunks | A13 satisfied; chunks preserve Q→A and table rows |
| **M3 — Embedding + store** | MiniLM-L6-v2 embeddings into ChromaDB | A11, A12 satisfied; re-run is idempotent |
| **M4 — Retrieval** | Top-k + scheme filter + field-type boost, similarity floor | A14 verified across all 5 schemes |
| **M5 — Generation + contract** | Grounded prompt, ≤ 3 sentences, one link, freshness line | A3, A4, A5 satisfied |
| **M6 — Guardrails** | Routing: advice / performance / PII / out-of-scope | A6, A7, A8, A9 satisfied |
| **M7 — UI** | Welcome line, 3 example questions, disclaimer, citation rendering | A15–A18 satisfied |
| **M8 — Docs & demo** | README, D4 sample Q&A, D5 disclaimer, demo recording | A19–A23 satisfied |

**Demo narrative (≤ 3 min):** ask a factual question and show the retrieved chunks → ask a second
question on a different scheme to show scoping → trigger an advice refusal → trigger a PII rejection →
show the "Last updated" and single-citation contract on screen.

---

## 16. Definition of done

The milestone is complete when every box in §14 is checked, all deliverables in §11 exist, and the
app runs end-to-end from a clean checkout with a single command. Any known deviation is written into the
README's **Known limits** section rather than left undocumented.
