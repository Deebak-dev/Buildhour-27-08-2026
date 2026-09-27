# Architecture — Mutual Fund FAQ Assistant (Facts-Only RAG Chatbot)

| Field | Value |
| --- | --- |
| Document | Software Architecture Document |
| Version | 1.0 (draft for review) |
| Date | 2026-09-27 |
| Derived from | `docs/PRD.md` v1.0 |
| Status | Awaiting approval before implementation |

> Every requirement in the PRD is mapped to a component in §18 (Traceability Matrix).
> Requirement IDs (`FR-n`, `NFR-n`, `A-n`) refer to `docs/PRD.md`.

---

## 1. Architecture drivers

The design is driven by five constraints, in priority order.

| # | Driver | Architectural consequence |
| --- | --- | --- |
| D1 | **No ungrounded answers** (NFR-1, FR-6.5) | Retrieval is the only source of truth. The LLM is a *formatter*, not a *knowledge source*. If retrieval yields nothing usable, the pipeline terminates in a refusal — it never falls through to generation. |
| D2 | **One valid citation per answer** (FR-6.3, NFR-2) | Citations are emitted by **code from chunk metadata**, never by the LLM. A URL allowlist derived from the ingested source list blocks hallucinated links. |
| D3 | **Advice and PII must be blocked before any generation** (FR-7.1–7.4) | A **pre-retrieval routing gate** sits in front of the whole pipeline. PII is detected by deterministic regex, not by the LLM, so it can never be talked past. |
| D4 | **Nothing sensitive persists** (FR-5.3, NFR-4) | PII is rejected at the edge. A single logging sink enforces a redaction policy; chat history is not persisted to disk. |
| D5 | **Every RAG stage must be demoable** (G6, FR-2.5) | Stages are separate modules with pure-ish interfaces and an individually runnable CLI, plus a trace object that makes retrieval inspectable in the UI. |

### 1.1 Design principles

1. **Fail closed, not open.** Uncertainty resolves to a refusal, never to a guess.
2. **Deterministic where possible.** Regex and allowlists guard the hard rules; the LLM only handles the fuzzy middle. This makes guardrails testable and free.
3. **Grounding is structural.** The generator cannot see anything except the retrieved chunk text — enforced by the prompt, the input contract, and the post-generation validator.
4. **Pipeline stages are idempotent.** Re-running ingestion rebuilds state rather than duplicating it.
5. **Point-in-time corpus.** The system answers as of `ingested_at`, and always says so.

---

## 2. System overview

The system has two flows that share a corpus and a vector store:

- **Flow A — Ingestion (offline, run once or on a schedule).** URLs → cleaned documents → chunks → vectors → ChromaDB.
- **Flow B — Query (online, per user question).** Query → guardrail routing → retrieval → grounded generation → validated answer → UI.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            OFFLINE  ·  FLOW A  ·  INGESTION                   │
│                                                                              │
│  sources.csv ──▶ ┌────────┐ ┌────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │
│  (5 schemes +   │ LOADING │▶│CHUNKING│▶│EMBEDDING │▶│  STORE   │▶│MANIFEST│ │
│   AMC/SEBI/AMFI) │ fetch + │ │struct- │ │ MiniLM-  │ │ ChromaDB │ │+hashes │ │
│                 │ clean + │ │ aware  │ │  L6-v2   │ │persist.  │ │+dates  │ │
│                 │ metadata│ │        │ │  384-d   │ │ local    │ │        │ │
│                 └────┬───┘ └───┬────┘ └──────────┘ └──────────┘ └────────┘ │
│                      │        │                                              │
│                 data/raw  data/chunks                                       │
│                 data/clean                                                │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   │  build-time only
                                   ▼
                        ┌─────────────────────┐
                        │  ChromaDB (local)   │◀──────┐
                        │  collection:        │       │ read-only at query time
                        │  mf_faq_chroma     │       │
                        │  + metadata + doc  │       │
                        └─────────────────────┘       │
                                   ▲                 │
┌──────────────────────────────────┴─────────────────┴───────────────────────────┐
│                            ONLINE   ·  FLOW B  ·  QUERY                       │
│                                                                              │
│  ┌──────┐   ┌───────────────┐   ┌──────────┐   ┌───────────┐   ┌──────────┐  │
│  │ UI   │──▶│  1. GUARDRAIL │──▶│ 2. RETRI │──▶│ 3. GENERA │──▶│ 4. VALI- │  │
│  │chat  │◀──│  route/reject │   │ EV(E)    │   │  TE       │◀──│ DATE     │  │
│  └──────┘   └───────────────┘   └──────────┘   └───────────┘   └──────────┘  │
│   §7           FR-7              FR-5          FR-6          FR-6.2–6.4    │
│  contract      PII/ADVICE/       top-k +       grounded      ≤3 sent ·     │
│  + 3 ex. Qs    PERF/OOS gate      scheme +      context-      1 link ·      │
│  + disclaimer                  field_type     only          freshness     │
│                                              boost                                      │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component inventory

| # | Component | Module | Stage | Implements |
| --- | --- | --- | --- | --- |
| C1 | Corpus manifest | `ingest/sources.csv` | — | FR-1.1, D2, A20 |
| C2 | Loader / cleaner | `ingest/load.py` | Loading | FR-1.2–1.6 |
| C3 | Chunker | `ingest/chunk.py` | Chunking | FR-2.1–2.5 |
| C4 | Embedder | `ingest/embed.py` | Embedding | FR-3.1–3.4 |
| C5 | Vector store writer | `ingest/store.py` | Vector store | FR-4.1–4.3 |
| C6 | Ingest orchestrator | `ingest/run_ingest.py` | Flow A | NFR-5, A10 |
| C7 | Guardrail router | `rag/guardrails.py` | Pre-retrieval | FR-7.1–7.5, A6–A9 |
| C8 | Redacting logger | `rag/logging_utils.py` | Cross-cutting | FR-5.3, NFR-4 |
| C9 | Retriever | `rag/retrieve.py` | Retrieval | FR-5.1–5.4, A14 |
| C10 | Prompt library | `rag/prompts.py` | Generation | FR-6.1–6.2 |
| C11 | Generator | `rag/answer.py` | Generation | FR-6.1–6.5 |
| C12 | Contract validator | `rag/validate.py` | Post-generation | FR-6.2–6.4, A3–A5 |
| C13 | Pipeline orchestrator | `rag/pipeline.py` | Flow B | A10 |
| C14 | Schemas / dataclasses | `rag/schemas.py` | — | Traceability |
| C15 | Config | `config.yaml` + `.env` | — | NFR-5, Q2 |
| C16 | UI | `app.py` | Presentation | FR-8.1–8.7, A15–A18 |
| C17 | Eval harness | `eval/run_eval.py` | — | A2, A6–A9, D4 |
| C18 | Test suite | `tests/` | — | NFR-1, NFR-2 |

---

## 4. Directory layout

```
buildhour-RAG/
├── app.py                        # C16 entry point (chat UI)
├── config.yaml                   # C15 non-secret config
├── .env.example                  # LLM API key placeholder
├── requirements.txt
├── Makefile                      # make ingest / run / eval
│
├── docs/
│   ├── Problemstatement.txt       # source brief
│   ├── PRD.md                    # requirements
│   ├── architecture.md           # this document
│   ├── README.md                 # D3
│   ├── source_list.csv           # D2 generated output
│   ├── sample_qa.md              # D4 generated output
│   └── disclaimer.md             # D5
│
├── ingest/                       # ── FLOW A ──────────────────────────
│   ├── sources.csv               # C1 input manifest (hand-curated)
│   ├── load.py                   # C2
│   ├── chunk.py                  # C3
│   ├── embed.py                  # C4
│   ├── store.py                  # C5
│   └── run_ingest.py             # C6 CLI: python -m ingest.run_ingest
│
├── rag/                          # ── FLOW B ──────────────────────────
│   ├── schemas.py                # C14 Document / Chunk / Route / Answer
│   ├── guardrails.py             # C7
│   ├── retrieve.py               # C9
│   ├── prompts.py                # C10
│   ├── answer.py                 # C11
│   ├── validate.py               # C12
│   ├── pipeline.py               # C13
│   └── logging_utils.py          # C8
│
├── data/                         # generated, git-ignored
│   ├── raw/                      # byte-faithful page snapshot
│   ├── clean/                    # extracted main text
│   ├── chunks/                   # chunk JSONL (inspectable)
│   ├── chroma/                   # persistent ChromaDB
│   └── manifest.json             # source_id → url, sha256, ingested_at
│
├── eval/
│   ├── golden_set.jsonl          # 10 factual queries + expected fields
│   ├── adversarial_set.jsonl     # advice / performance / PII / OOS
│   └── run_eval.py               # C17
│
├── tests/                        # C18
└── logs/                         # redacted only (C8 enforces)
```

---

## 5. Flow A — Ingestion pipeline

### 5.1 Sequence

```
run_ingest.py
     │
     ▼
┌──────────────┐  read + validate   ┌──────────────────────────────┐
│ sources.csv  │───────────────────▶│ C2 load.fetch()              │
│  12+ rows    │                    │  • robots/UA + timeout        │
└──────────────┘                    │  • HTTP status recorded      │
                                    │  • → data/raw/{source_id}.html│
                                    │  • C2 load.extract()         │
                                    │    - strip nav/footer/ads     │
                                    │    - main-content heuristics  │
                                    │    - → data/clean/{sid}.txt   │
                                    │  • FR-1.5 failure → log, no  │
                                    │    empty doc                   │
                                    └───────────────┬──────────────┘
                                                    ▼
                                    ┌──────────────────────────────┐
                                    │ C3 chunk.build_chunks()      │
                                    │  • detect shape per doc      │
                                    │    (FAQ / table / narrative) │
                                    │  • Q→A pair = 1 chunk        │
                                    │  • table row + header = 1    │
                                    │  • narrative → 400–800 tok   │
                                    │  • tag field_type            │
                                    │  • → data/chunks/chunks.jsonl│
                                    └───────────────┬──────────────┘
                                                    ▼
                                    ┌──────────────────────────────┐
                                    │ C4 embed.embed_chunks()     │
                                    │  all-MiniLM-L6-v2 (384-d)   │
                                    │  normalize=True             │
                                    │  batched (32)               │
                                    └───────────────┬──────────────┘
                                                    ▼
                                    ┌──────────────────────────────┐
                                    │ C5 store.upsert()           │
                                    │  PersistentClient(data/chroma)
                                    │  collection: mf_faq_chroma  │
                                    │  ids = sha256(chunk_text)   │
                                    │  → idempotent re-run        │
                                    └───────────────┬──────────────┘
                                                    ▼
                                    ┌──────────────────────────────┐
                                    │ C6 write manifest.json +    │
                                    │    docs/source_list.csv     │
                                    │    (D2 deliverable, A20)     │
                                    └──────────────────────────────┘
```

### 5.2 Data model

**Document** (`rag/schemas.py`)

| Field | Type | Source | Notes |
| --- | --- | --- | --- |
| `source_id` | str | derived | stable slug, e.g. `hdfc-large-cap__groww_scheme` |
| `scheme` | str | manifest | canonical scheme key, one of the 5 |
| `category` | str | manifest | `large_cap`, `flexi_cap`, `elss`, `small_cap`, `hybrid` |
| `doc_type` | str | manifest | `scheme_page` \| `factsheet` \| `kim` \| `sid` \| `faq` \| `fees` \| `riskometer` \| `benchmark` \| `statement_guide` |
| `title` | str | page `<h1>` | |
| `publisher` | str | manifest | `hdfc_amc` \| `sebi` \| `amfi` \| `groww` (see ADR-4) |
| `source_url` | str | manifest | the citation URL |
| `retrieved_at` | ISO-8601 | loader | drives the "Last updated from sources" line |
| `sha256` | str | loader | integrity + idempotency |
| `http_status` | int | loader | FR-1.5 |
| `content_chars` | int | cleaner | thin-page detection (R2) |
| `raw_path` / `clean_path` | str | loader | auditability (FR-1.4) |

**Chunk** — a `Document` plus:

| Field | Type | Notes |
| --- | --- | --- |
| `chunk_id` | str | `sha256(source_id + ordinal)` |
| `text` | str | the embedded unit |
| `section` | str | heading path, e.g. `"Fees & charges > Direct plan"` |
| `field_type` | str \| null | controlled vocabulary, see §5.3 |
| `ordinal` | int | position within document |

**Chroma collection `mf_faq_chroma`**

| Aspect | Value |
| --- | --- |
| Embedding function | `sentence-transformers/all-MiniLM-L6-v2` (384-d) |
| Distance | cosine (via normalized embeddings) |
| IDs | `chunk_id` (stable → upsert-safe) |
| Metadata | `scheme`, `category`, `doc_type`, `field_type`, `section`, `source_url`, `publisher`, `retrieved_at`, `ordinal` |
| Embedding config | persisted in collection metadata so a model change forces a rebuild (FR-3.3) |

### 5.3 `field_type` controlled vocabulary

`field_type` is what lets a "expense ratio" query prefer a fee chunk over a narrative one (FR-5.3), and
it is the main defence against MiniLM confusions such as *regular vs direct plan* (R4).

```
expense_ratio | exit_load | min_sip | min_lumpsum | lock_in | tax_benefit
riskometer | benchmark | nav | fund_manager | scheme_type | direct_vs_regular
document_download | statement_guide | tax_report | other
```

Assigned at chunk time by rules over section headings and the text (keyword + regex), not by an LLM —
cheap, deterministic, and inspectable.

### 5.4 Chunking strategy (per FR-2.1, decided from the data)

`chunk.py` **profiles each document first**, then picks a strategy. The decision and the resulting
parameters are written into `data/manifest.json` and quoted in the README (FR-2.5).

| Detected shape | Strategy | Params |
| --- | --- | --- |
| FAQ Q→A blocks | one Q→A pair per chunk, never split | pair-boundary split |
| Fee / holdings tables | one row + its column header row per chunk | header repeated per chunk |
| KIM / SID narrative | heading-anchored recursive split | 400–800 tokens, 12% overlap |
| Unstructured prose | recursive character split on paragraph → sentence | 400–800 tokens, 12% overlap |

**Rejected by design:** fixed-size character splitting — it destroys the Q→A and table-row units that
every fact lookup depends on (see PRD §9.1).

Tuning loop, run once and recorded: build chunks → run `eval/run_eval.py` on the golden set → if
recall@5 < target, adjust size/overlap → repeat. The winning config lands in `config.yaml`.

### 5.5 Idempotency and rebuild (FR-4.2, NFR-5)

- `chunk_id` is content-derived → re-ingesting the same page overwrites, never duplicates.
- `--rebuild` drops and recreates the collection (required whenever the embedding model or dimension changes).
- `manifest.json` stores the embedding model + dimension; `run_ingest` refuses to write into a
  collection whose stored dimension differs.

---

## 6. Flow B — Query pipeline

### 6.1 Sequence

```
User query (plain text)
        │
        ▼
┌────────────────────────────────────────────────────────────────┐
│ STAGE 0 — EDGE: PII SCREEN (deterministic, no LLM)            │
│   rag/guardrails.py :: detect_pii(text)                        │
│   hit → REJECT. Return neutral message. **Discard text.**      │
│   Never log raw text. Never persist. (FR-7.4, D4)             │
└───────────────────────────┬────────────────────────────────────┘
                            │ clean
                            ▼
┌────────────────────────────────────────────────────────────────┐
│ STAGE 1 — ROUTE (C7)                                          │
│   1a. LLM classifier → FACTUAL | ADVICE | PERFORMANCE | OOS     │
│   1b. keyword reinforcement (deterministic override)            │
│   precedence: PII > PERFORMANCE > ADVICE > OOS > FACTUAL        │
└───┬──────────────┬───────────────┬──────────────┬─────────────┘
    │ADVICE        │PERFORMANCE     │OOS           │FACTUAL
    ▼              ▼               ▼              ▼
 polite         factsheet        "not in my    ┌──────────────────────┐
 facts-only      link, no         sources"      │ STAGE 2 — RETRIEVE   │
 msg + edu      figures                         │  C9                  │
 link                                        │  a. scheme resolve    │
 (FR-7.2)      (FR-7.3)        (FR-7.5)      │  b. metadata filter  │
                                                  │  c. top-k = 10       │
                                                 │  d. field_type boost │
                                                 │  e. similarity floor │
                                                 └──────────┬───────────┘
                                                            │ chunks
                                                            ▼
                                                 ┌──────────────────────┐
                                                 │ STAGE 3 — GENERATE   │
                                                 │  C10 + C11           │
                                                 │  context = chunks    │
                                                 │  ONLY. ≤3 sentences. │
                                                 │  → structured JSON   │
                                                 └──────────┬───────────┘
                                                            ▼
                                                 ┌──────────────────────┐
                                                 │ STAGE 4 — VALIDATE   │
                                                 │  C12 (hard gates)    │
                                                 │  • sentence count ≤3 │
                                                 │  • exactly 1 URL     │
                                                 │  • URL ∈ allowlist   │
                                                 │  • no advice/return  │
                                                 │    vocabulary        │
                                                 │  • figures verbatim  │
                                                 │  FAIL → retry once   │
                                                 │        → else refuse │
                                                 │  C12 appends         │
                                                 │  "Last updated from  │
                                                 │   sources: <date>"   │
                                                 └──────────┬───────────┘
                                                            ▼
                                                 ┌──────────────────────┐
                                                 │ UI render (C16)      │
                                                 │ answer · 1 citation  │
                                                 │ · freshness line     │
                                                 │ · [chunks] expander  │
                                                 └──────────────────────┘
```

### 6.2 Stage 0 — PII screen (deterministic, runs first)

Because this is a hard regulatory-style rule, it must not depend on an LLM's judgement (D3).

| Class | Detection |
| --- | --- |
| PAN | `[A-Z]{5}[0-9]{4}[A-Z]` |
| Aadhaar | 12 digits, first digit 2–9, optional spaces/dashes, **plus Verhoef checksum** to cut false positives |
| Account / folio number | 6–12 consecutive digits **with** a keyword (`account`, `folio`, `demat`, `pan linked`) |
| OTP | 4–6 digits **with** an `otp`/`one time password` keyword |
| Email | standard RFC-style pattern |
| Phone | `(+91)?[6-9]\d{9}` and 10-digit variants |

Two safeguards make this reliable:

1. **Keyword-gated patterns.** A bare 8-digit number is not an account number. The keyword requirement
   keeps legitimate figures (SIP amounts, percentages, lock-in years) from being false-positived.
2. **Known-safe numeric contexts.** A number immediately preceded by `%`, `₹`, `Rs`, `min`, `SIP`, or
   followed by `SIP`/`%`/`years` is whitelisted.

On a hit: return the §7 PII response, **drop the input**, log only `{route, matched_class}` — never the text.

### 6.3 Stage 1 — Router

Hybrid design: LLM classifier for recall, keyword rules for precision (D2 principle 2).

| Route | Deterministic triggers (examples) | LLM label |
| --- | --- | --- |
| `PII` | §6.2 regex | n/a (pre-LLM) |
| `PERFORMANCE` | return, returns, CAGR, performance, how much did it grow, best performing, ranking, top performer, NAV trend, X-year return | `PERFORMANCE` |
| `ADVICE` | should I, which is best, recommend, suggest, worth buying, good time, is it safe for me, suitable for, my portfolio, allocate, target, better than | `ADVICE` |
| `OOS` | today's NAV, current price, live quote, app login, account opening, and any scheme outside the 5 | `OUT_OF_SCOPE` |
| `FACTUAL` | fallback | `FACTUAL` |

Precedence is deliberate: `PERFORMANCE` beats `ADVICE` so that "which fund has the best returns and
should I buy it" returns the factsheet redirect rather than a generic refusal.

A `FACTUAL` label is **not** trusted blindly — Stage 2/4 gates still apply, so a misclassification
cannot produce an ungrounded answer.

### 6.4 Stage 2 — Retrieval (C9)

| Step | Behaviour | Rationale |
| --- | --- | --- |
| a. Scheme resolution | match the query against the 5 canonical scheme names; extract an explicit `scheme` filter | cross-scheme contamination is the top correctness risk (R4, A14) |
| b. Filter | `where={"scheme": ...}` when resolved | scopes the candidate set |
| c. Search | `n_results = k + 2` (over-fetch, then post-boost) | headroom for re-ranking |
| d. `field_type` boost | `+0.05` cosine-equivalent to chunks whose `field_type` matches the query intent | FR-5.3; guards *direct vs regular* confusions |
| e. Similarity floor | `score < tau` (baseline `0.35`, tuned on the golden set) → treat as no-answer | FR-5.1, A9 |
| f. Return | top `k = 5` chunks + scores + a `RetrievalTrace` | feeds the UI expander (FR-8.6) |

An unresolvable scheme is **not** an error — the query proceeds unscoped, so general questions
("how do I download a capital gains statement") still work.

**Hybrid search (stretch, FR-5.4 / Q5):** BM25 over chunk text ∪ dense vectors, merged by reciprocal
rank fusion, optional cross-encoder re-ranker. Behind a `retrieval.mode` config flag so the MVP path
stays simple and the demo can show a before/after.

### 6.5 Stage 3 — Generation (C10, C11)

The system prompt's load-bearing clauses (full text in `rag/prompts.py`):

```
You answer questions about HDFC mutual fund schemes using ONLY the
CONTEXT below. The context is the complete source of truth.

Hard rules:
1. Use only facts present in CONTEXT. If a fact is not in CONTEXT, say you
   do not have it in your sources. Never use prior knowledge.
2. Answer in at most 3 sentences. Factual, neutral, plain English.
3. Never recommend, compare, rank, or rate schemes. Never say "you should".
4. Never state or compute returns, CAGR, NAV movement, or projections. If
   asked, say the official factsheet has the figures and link it.
5. Copy numbers, rates, and names VERBATIM from CONTEXT. Do not round,
   reformat, or recompute.
6. Return JSON only: {"answer": str, "grounded_field": str|null}
```

Design points:

- **Structured output** (`{"answer", "grounded_field"}`) so the validator has a stable field to
  inspect, and so the citation is attached by code rather than generated (D2).
- **The model never sees the user query alone with the schema** — it receives context-only inputs on
  the factual path, which makes ungrounded generation structurally hard (D1).
- **`grounded_field`** is the `field_type` the answer used. If it does not match the query intent, the
  validator downgrades the response to a "not in my sources" reply.

### 6.6 Stage 4 — Contract validator (C12)

The last line of defence. Every gate is a hard check on the generated answer; **any failure means the
answer is discarded**, not patched.

| Gate | Check | On failure |
| --- | --- | --- |
| G1 Structure | valid JSON, non-empty `answer` | retry once → refuse |
| G2 Length | sentence count ≤ 3 | retry with an explicit length instruction → refuse |
| G3 Citation | exactly one URL, and it parses | refuse |
| G4 Allowlist | URL ∈ `manifest.json` source set | **refuse** (blocks hallucinated links, NFR-2) |
| G5 Vocabulary | no advice/return/performance terms present | retry once → refuse |
| G6 Figure fidelity | every number in the answer appears in the context chunk | **refuse** (blocks R7 drift) |
| G7 Freshness | `Last updated from sources:` line present | **appended by code** from `retrieved_at` |
| G8 Grounding | `grounded_field` matches query intent, or the answer states it lacks the data | downgrade to not-in-sources |

Two retries maximum, then refusal — this bounds latency (NFR-3) while keeping the strictness.

### 6.7 The §7 response contract, implemented

`validate.py` renders the final payload; `app.py` only displays it.

```
<answer ≤ 3 sentences>

Source: <one URL from the winning chunk's metadata>
Last updated from sources: <DD MMM YYYY, from chunk.retrieved_at>
```

The `Source:` and `Last updated from sources:` lines are **constructed in Python from metadata**. The
LLM contributes only the answer prose. This is the single most important implementation decision in the
project: it makes the citation and freshness guarantees impossible to violate (D2, D1).

---

## 7. UI architecture (C16)

### 7.1 Screen structure

| Element | Content | Requirement |
| --- | --- | --- |
| Header | Product name + corpus scope chip (`1 AMC · 5 schemes · N docs · ingested <date>`) | FR-9.1, A15 welcome line |
| Disclaimer banner | The §11.3 disclaimer, always visible | FR-8.3, A17, D5 |
| Example questions | **Exactly 3**, clickable, one per intent | FR-8.2, A16 |
| Chat thread | User bubble → answer + citation + freshness | FR-8.4, A8.5 |
| Retrieved-chunks expander | Collapsed by default: chunk text, score, source, `field_type` | FR-8.6, G6 |

The 3 example questions map to the three headline behaviours, which is also the demo script:

1. "What is the expense ratio of HDFC Large Cap Fund – Direct Growth?" → cited factual answer
2. "What is the lock-in period for HDFC ELSS Tax Saver Fund?" → cited factual answer
3. "Should I buy HDFC Small Cap Fund?" → polite facts-only refusal

### 7.2 State and persistence

- Session state is **in-memory only**. No user input is written to disk (FR-5.3, NFR-4).
- `logs/` receives only redacted structured events: `{ts, route, scheme, n_chunks, top_score, latency_ms, outcome}`.
- No admin, back-end, or configuration screens are exposed (FR-8.7, A18).

---

## 8. Configuration (C15)

`config.yaml` (non-secret, committed) and `.env` (secrets, git-ignored).

```yaml
corpus:
  manifest: ingest/sources.csv
  raw_dir: data/raw
  clean_dir: data/clean

chunking:
  strategy: auto            # auto | faq_pair | table_row | narrative
  target_tokens: 600
  min_tokens: 200
  max_tokens: 800
  overlap_pct: 12

embedding:
  model: sentence-transformers/all-MiniLM-L6-v2
  dimension: 384
  normalize: true
  batch_size: 32

vectorstore:
  backend: chromadb
  persist_dir: data/chroma
  collection: mf_faq_chroma
  rebuild_on_dimension_change: true

retrieval:
  mode: vector              # vector | hybrid
  top_k: 10                 # the LLM context window
  over_fetch: 10
  similarity_floor: 0.35    # tuned on golden set
  field_type_boost: 0.35
  field_misfit_penalty: 0.15
  scheme_filter: strict

generation:
  provider: ${LLM_PROVIDER} # pluggable, one env var
  model: ${LLM_MODEL}
  temperature: 0.0          # determinism for the demo
  max_sentences: 3
  max_retries: 2
  structured_output: true

guardrails:
  pii_screen: enabled
  allowlist_from_manifest: true
  log_raw_queries: false

answer:
  freshness_label: "Last updated from sources:"
  date_format: "%d %b %Y"
```

**Q2 resolved by design:** `provider` and `model` are env-driven, so the LLM can be swapped without
touching pipeline code (D5, §10 of the PRD).

---

## 9. Interface contracts

Signatures the modules expose — the seams the team codes against.

```python
# rag/schemas.py
@dataclass(frozen=True)
class Document:  source_id, scheme, category, doc_type, title, publisher,
                 source_url, retrieved_at, sha256, http_status, content_chars,
                 raw_path, clean_path
@dataclass(frozen=True)
class Chunk:     chunk_id, source_id, scheme, field_type, section, ordinal, text
@dataclass(frozen=True)
class RouteResult: label, reason, matched_classes, scheme_hint
@dataclass
class RetrievalTrace: query, scheme_filter, chunks: list[ScoredChunk], tau, top_score
@dataclass
class AnswerPayload: text, source_url, last_updated, grounded_field,
                     route, trace, degraded: bool, validators_passed: list[str]

# ingest/load.py
def fetch(manifest: list[ManifestRow], *, timeout: int = 30) -> list[Document]
def extract(doc: Document) -> tuple[str, int]          # (clean_text, n_chars)

# ingest/chunk.py
def profile(text: str) -> DocShape                    # FAQ | TABLE | NARRATIVE
def build_chunks(docs: list[Document]) -> list[Chunk]
def tag_field_type(text: str, section: str) -> str | None

# ingest/embed.py
def embed_chunks(chunks: list[Chunk], cfg: EmbedCfg) -> list[list[float]]

# ingest/store.py
def open_collection(cfg: StoreCfg, *, rebuild: bool = False) -> Collection
def upsert(collection: Collection, chunks: list[Chunk], vectors: list[list[float]]) -> int
def url_allowlist() -> set[str]                       # from manifest.json

# rag/guardrails.py
def detect_pii(text: str) -> list[str]                # matched classes, [] if clean
def route(query: str, *, llm: LLM, allowlist: set[str]) -> RouteResult

# rag/retrieve.py
def retrieve(query: str, *, collection: Collection, cfg: RetrievalCfg) -> RetrievalTrace

# rag/answer.py
def generate(query: str, trace: RetrievalTrace, *, llm: LLM) -> DraftAnswer

# rag/validate.py
def validate(draft: DraftAnswer, query: str, trace: RetrievalTrace,
             allowlist: set[str], cfg: GenCfg) -> AnswerPayload

# rag/pipeline.py
def ask(query: str) -> AnswerPayload                  # Flow B, all stages

# app.py
streamlit_ui(ask_fn=ask) -> None
```

Every one of these is pure-ish: `(input, config) → output`, no hidden global state, no I/O except the
two places that must touch disk (`load.fetch`, `store.upsert`). That is what makes A10 and the unit tests
possible.

---

## 10. Error handling and degradation

| Failure | Detection | Behaviour |
| --- | --- | --- |
| Source page unreachable / non-200 | `http_status` in loader | Log + record `status=failed` in D2; continue with remaining sources. Never write an empty document. (FR-1.5) |
| JS-rendered page yields no text | `content_chars < 200` | Flag in manifest; substitute the AMC-hosted primary doc for the same fact. (R2) |
| Primary + fallback both missing a fact | — | "not in my sources" — stated, not guessed. (A9) |
| Vector store empty / collection missing | Chroma lookup | Fail fast at startup with a run-to-`make ingest` message. Never silently return no results. |
| Embedding dimension mismatch | `collection.metadata` vs `cfg.dimension` | Block ingestion; require `--rebuild`. (FR-4.2) |
| LLM timeout / error | client exception | One retry, then a graceful "I couldn't answer right now" — **never** an ungrounded fallback. (NFR-1) |
| LLM returns malformed JSON | `json.loads` in G1 | Retry with a stricter instruction, then refuse. |
| Validator gate fails | G2/G5 | Retry once, then refuse. |
| Latency over budget | timing | Log; no behavioural change. (NFR-3) |

**Universal rule:** every failure path terminates in a *safe* output (refusal / "not in my sources" /
retry), never in a fabricated answer.

---

## 11. Observability (C8)

One redacting sink, used by every module.

- **Structured events only:** `{ts, stage, route, scheme, n_chunks, top_score, validators_passed, latency_ms, outcome, degraded}`.
- **Never logged:** raw user query text, any PII, full chunk bodies, or API keys.
- **Trace per request:** `RetrievalTrace` is returned to the UI expander (FR-8.6) and to the eval
  harness — this is what makes the RAG internals visible in the ≤ 3-min demo.
- **Rejection log:** PII events log `{route, matched_classes}` only, so the block rate is auditable
  (A8) without ever storing the offending text.

---

## 12. Testing strategy (C17, C18)

| Suite | What it asserts | Maps to |
| --- | --- | --- |
| `test_guardrails.py` | each PII class is caught; the §6.2 whitelists do **not** false-positive on `₹500`, `1.5%`, `3 years`, `80C` | A8, R6 |
| `test_chunking.py` | Q→A pairs are never split; table rows carry their headers; all chunks within token bounds | FR-2.3, A13 |
| `test_contract.py` | ≤ 3 sentences; exactly one URL; URL ∈ allowlist; freshness line present | A3, A4, A5 |
| `test_no_pii_logs.py` | no PII pattern appears in any file under `logs/` after a PII query | FR-5.3, A8 |
| `test_grounding.py` | a fabricated figure in a draft is rejected by G6 | NFR-1, R7 |
| `test_scoping.py` | a Large Cap query never returns a Small Cap chunk | A14, R4 |
| `golden_set.jsonl` | 10 factual queries → expected `field_type` + scheme; measures recall@k and citation validity | A2, D4 |
| `adversarial_set.jsonl` | 10+ advice / performance / PII / OOS queries → expected route | A6, A7, A8, A9 |

`make eval` prints a pass-rate table. The golden + adversarial sets double as the source of the D4
sample Q&A file, so evaluation and the deliverable are produced by the same run (A22).

---

## 13. Security and privacy

| Control | Implementation |
| --- | --- |
| No PII ingress | Deterministic screen before any processing; input discarded on hit (FR-7.4) |
| No PII at rest | In-memory chat only; redacting logger; no query-text persistence (NFR-4) |
| Least-privilege egress | Exactly two outbound destinations: the crawl at ingest, and the configured LLM endpoint at query time. No analytics, no telemetry. |
| No user accounts | No auth, no session cookies, no third-party identity (FR-5.3) |
| Secret handling | API keys in `.env`, git-ignored; never logged, never in the UI |
| Prompt-injection surface | The only untrusted text reaching the model is page content. Extraction strips scripts and hidden elements, and the prompt forbids following instructions found in CONTEXT. |
| Link safety | Citations are allowlisted to the ingested manifest and rendered as plain external links with `rel="noopener noreferrer"` (G4) |

---

## 14. Deployment and run modes

| Mode | Command | Use |
| --- | --- | --- |
| Ingest (offline) | `python -m ingest.run_ingest [--rebuild]` | M1–M3, refreshing the corpus |
| Local app | `streamlit run app.py` | Daily use, demo recording |
| Headless CLI | `python -m rag.pipeline --q "..."` | Debugging, eval, CI |
| Eval | `python -m eval.run_eval` | Generates D4 + pass-rate table |
| Tests | `pytest tests/` | Pre-commit gate |

All modes are CPU-only, Windows + Linux compatible (NFR-6). A single `make setup && make ingest && make run`
sequence reproduces everything from a clean checkout (NFR-5, A19).

---

## 15. Extension points

Deliberate seams, so scope growth does not require rewrites:

| Extension | Where it plugs in | Cost |
| --- | --- | --- |
| More schemes / AMCs | add rows to `ingest/sources.csv`, extend the scheme registry in `schemas.py` | ingest only |
| Primary AMC/SEBI sources for R1 | new manifest rows with `publisher=hdfc_amc|sebi|amfi`; prefer them in `retrieve.py` | low |
| Hybrid BM25 + re-ranker | `retrieval.mode=hybrid` in `config.yaml` (FR-5.4, Q5) | medium |
| A different LLM provider | `generation.provider/model` env vars only (Q2) | near zero |
| A different embedding model | `embedding.model` + dimension, then `--rebuild` (FR-3) | re-embed |
| A non-Chroma vector DB | behind `store.open_collection` | medium |
| Re-ingestion on a schedule | wrap `run_ingest.py` in a cron/task | low |
| Multilingual | add a translation step before routing; requires corpus expansion | high |

---

## 16. Architecture decision log (ADRs)

| ID | Decision | Rationale | Alternatives rejected |
| --- | --- | --- | --- |
| ADR-1 | Citations and the freshness line are emitted by **code from chunk metadata**, not by the LLM | Makes the one-citation and freshness guarantees structurally impossible to violate (D2) | Letting the model write the source line — a hallucination path |
| ADR-2 | A **deterministic PII screen runs before the LLM** | A hard privacy rule must not depend on model judgement; free and unit-testable (D3) | LLM-based PII detection — non-deterministic, bypassable, paid |
| ADR-3 | The generator receives **context only** and returns **structured output** | Enforces grounding structurally rather than by instruction alone (D1) | Free-form generation + post-hoc fact-checking — unreliable |
| ADR-4 | Ingest both the Groww seed pages **and** the primary AMC/SEBI/AMFI documents; prefer the primary citation | Resolves the R1 source-authority conflict while following the brief's URLs (R1, Q1) | Groww-only citations — a likely mentor challenge |
| ADR-5 | Structure-aware chunking chosen **after profiling the real corpus** | The brief requires a data-driven decision; fixed-size splitting breaks Q→A and table rows (FR-2.1) | Fixed 1000-char chunks — simple, but loses fact units |
| ADR-6 | `field_type` tagging by rules, boosting at retrieval time | Directly counters the highest-severity risk: MiniLM blurring *direct vs regular* and *exit-load tiers* (R4) | Relying on embeddings alone |
| ADR-7 | ChromaDB persistent local client, **content-derived chunk IDs** | Zero-setup, free, and idempotent re-ingestion (FR-4, NFR-5) | Pinecone/Qdrant cloud — account and cost for a class demo |
| ADR-8 | `all-MiniLM-L6-v2` (384-d) | Specified by the brief; small, fast, runs on CPU (FR-3) | Larger embedders — better quality, slower on CPU |
| ADR-9 | Structured validator with **max 2 retries, then refuse** | Bounded latency while keeping guardrails strict (NFR-3) | Unbounded retry loops; silent patching of bad output |
| ADR-10 | A retrieval trace surfaced in the UI expander | Turns the RAG internals into visible demo material (G6, FR-8.6) | Opaque answers — a weaker demo |
| ADR-11 | LLM provider is env-configurable, not hardcoded | The brief fixes embeddings and the vector DB but not the LLM (Q2) | Hardcoding one vendor |

---

## 17. Assumptions

1. Seed pages are publicly reachable and their text is extractable; where not, the AMC-hosted primary
   document substitutes (R2).
2. A free-tier LLM endpoint is available and reachable from the demo machine (Q2).
3. The 5 schemes' public pages publish the fact types in §4.4 of the PRD.
4. The corpus is a **point-in-time snapshot**; facts are correct as of `ingested_at` and the UI says so (R5).
5. `similarity_floor = 0.35` and `field_type_boost = 0.05` are **starting values** to be tuned on the
   golden set, then frozen in `config.yaml` (§5.4).
6. §5.3's `field_type` vocabulary covers the fact types in scope; unknown types fall back to `other`.

---

## 18. Traceability matrix

### 18.1 Functional requirements

| PRD req | Component(s) | Section | Acceptance |
| --- | --- | --- | --- |
| FR-1.1 Crawl 5 schemes + primary docs | C1, C2 | §5.1 | A1, A20 |
| FR-1.2 Extract main text, strip noise | C2 | §5.1 | A1 |
| FR-1.3 Document metadata | C2, C14 | §5.2 | A1 |
| FR-1.4 Raw + cleaned snapshot | C2 | §4, §5.2 | A21 |
| FR-1.5 Log fetch failures | C2, C8 | §10 | A21 |
| FR-1.6 Record retrieval timestamp | C2 | §5.2 | A5 |
| FR-2.1 Data-driven chunking decision | C3 | §5.4 | A13 |
| FR-2.2 Default 400–800 tokens, 10–15% overlap | C3 | §5.4 | A13 |
| FR-2.3 Keep Q→A and table rows intact | C3 | §5.4 | A13 |
| FR-2.4 `field_type` tagging | C3 | §5.3 | A14 |
| FR-2.5 Document the strategy | C3, C6 | §5.4 | A21 |
| FR-3.1 MiniLM-L6-v2 | C4 | §5.2 | A11 |
| FR-3.2 384 dimensions | C4, C15 | §5.2, §8 | A11 |
| FR-3.3 Store model + dim in metadata | C5 | §5.2 | A11 |
| FR-3.4 Local/CPU embedding | C4 | §2 | A11 |
| FR-4.1 ChromaDB persistent | C5 | §5.2 | A12 |
| FR-4.2 Explicit collection, idempotent | C5 | §5.5 | A12 |
| FR-4.3 Metadata filtering | C5, C9 | §6.4 | A14 |
| FR-5.1 top-k + similarity floor | C9 | §6.4 | A9 |
| FR-5.2 Scheme filter | C9 | §6.4 | A14 |
| FR-5.3 `field_type` boost | C9 | §6.4 | A2 |
| FR-5.4 Hybrid + re-ranker (stretch) | C9 | §6.4 | — |
| FR-6.1 Context-only grounding | C10, C11 | §6.5 | A2, A3 |
| FR-6.2 ≤ 3 sentences, neutral tone | C10, C12 | §6.5, §6.6 | A4 |
| FR-6.3 Exactly one source link | C12 | §6.6, §6.7 | A3 |
| FR-6.4 `Last updated from sources:` | C12 | §6.6, §6.7 | A5 |
| FR-6.5 Insufficient context → say so | C9, C11 | §6.4, §10 | A9 |
| FR-6.6 Prompt + contract in one module | C10, C12 | §6.5 | A21 |
| FR-7.1 Pre-retrieval routing | C7 | §6.1, §6.3 | A6–A9 |
| FR-7.2 Advice → polite msg + edu link | C7 | §6.3 | A6 |
| FR-7.3 Performance → factsheet link, no figures | C7 | §6.3 | A7 |
| FR-7.4 PII rejected pre-retrieval, not stored | C7, C8 | §6.2 | A8 |
| FR-7.5 Out-of-corpus → not in sources | C7, C9 | §6.3, §6.4 | A9 |
| FR-8.1 Welcome line | C16 | §7.1 | A15 |
| FR-8.2 Exactly 3 example questions | C16 | §7.1 | A16 |
| FR-8.3 "Facts-only. No investment advice." visible | C16 | §7.1 | A17 |
| FR-8.4 Render answer + link + freshness | C12, C16 | §6.7, §7.1 | A3, A5 |
| FR-8.5 Link visible, not hidden | C16 | §7.1 | A3 |
| FR-8.6 Optional chunks expander | C9, C16 | §7.1 | — |
| FR-8.7 No back-end screens/screenshots | C16 | §7.2 | A18 |
| FR-9.1 State corpus scope + ingest date | C16 | §7.1 | A9 |
| FR-9.2 Disclaimer as single constant | C16 | §7.1 | A23 |

### 18.2 Non-functional requirements

| PRD req | Mechanism | Section |
| --- | --- | --- |
| NFR-1 Zero ungrounded answers | context-only generation + floor + validator | §6.4–6.6 |
| NFR-2 Citation integrity | allowlist gate G4 | §6.6 |
| NFR-3 Latency ≤ 10 s | k=5, over-fetch 2, max 2 retries | §6.4, §6.6 |
| NFR-4 No PII in logs | redacting sink, no query persistence | §7.2, §11 |
| NFR-5 Reproducibility | idempotent ingest, `make` targets | §5.5, §14 |
| NFR-6 Windows + Linux, CPU-only | pure-Python stack, no GPU | §14 |
| NFR-7 Maintainable stages | pure-ish interfaces per stage | §9 |
| NFR-8 Transparency | chunking rationale + config in README | §5.4, §8 |
| NFR-9 Offline embedding | local MiniLM, local Chroma | §2 |

### 18.3 Risks → architectural controls

| PRD risk | Control | Section |
| --- | --- | --- |
| R1 Source authority mismatch | ADR-4 dual-source ingest, primary-preferred citation | §16 |
| R2 JS-rendered pages | `content_chars` thin-page detection + AMC fallback | §5.2, §10 |
| R3 Narrow corpus | explicit "not in my sources" + UI scope note | §6.4, §7.1 |
| R4 MiniLM confusions | strict scheme filter, `field_type` boost, floor, `test_scoping` | §5.3, §6.4, §12 |
| R5 Stale facts | mandatory freshness line, ingest date in UI and D2 | §5.2, §6.7 |
| R6 Guardrail bypass | routing first + adversarial eval set | §6.3, §12 |
| R7 Number drift | context-only + figure-fidelity gate G6 | §6.5, §6.6 |
| R8 Crawl failures | raw snapshot cache, idempotent re-ingest | §5.5, §10 |

---

## 19. Build order

Sequence follows the PRD milestones so that each phase is independently demoable.

| Phase | Build | Gate |
| --- | --- | --- |
| M0 | `sources.csv`, scheme registry, config skeleton | Corpus + source policy agreed (Q1) |
| M1 | C2 loader + `manifest.json` + D2 | Every manifest row has content or a logged failure |
| M2 | C3 chunker + `chunks.jsonl` | `test_chunking` green; rationale documented |
| M3 | C4 + C5 | `test_scoping` runs; re-ingest is idempotent |
| M4 | C9 retriever + `RetrievalTrace` | recall@k measured on the golden set |
| M5 | C10 + C11 + C12 | A3, A4, A5 pass |
| M6 | C7 guardrails + C8 logger | A6–A9 pass, incl. `test_no_pii_logs` |
| M7 | C16 UI | A15–A18 pass |
| M8 | C17 eval + docs D3/D4/D5 + demo recording | A19–A23 pass |

**Demo script (≤ 3 min):** ask question 1 with the chunks expander open → ask a question on a second
scheme to show scoping → trigger the advice refusal → trigger the PII rejection → show the one-citation
and "Last updated from sources" contract on screen.
