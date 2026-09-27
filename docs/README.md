# HDFC Mutual Fund Facts Assistant

A retrieval-augmented assistant that answers factual questions about five HDFC
Mutual Fund schemes from their published scheme pages, and refuses everything
else. It cites the single source behind every fact it states, and it will not
recommend, compare, rate, or project.

> **Facts-only. No investment advice.**

---

## Measured results

From `python -m eval.run_eval` on this machine, at run
`2026-09-27T10:55:45+00:00`, against provider `groq` /
`qwen/qwen3.8-27b`. Every number below is read from
`eval/results.json`; none of it is hand-written.

| Suite | Result |
|---|---|
| Adversarial (routing + PII) | **15/15 (100.0%)** |
| Adversarial retrieval checks only | 20/20 (100.0%) |
| Golden (factual contract) | **5/10 (50.0%)** |
| Overall | **20/25 (80.0%)** |
| Golden retrieval checks only | **24/29 (82.8%)** |
| Golden generation checks only | **40/49 (81.6%)** |
| Rows that refused instead of answering | 2 of 25 |
| Rows that failed from a provider outage | 0 of 25 |
| recall@1 | 6/10 (60%) |
| recall@3 / @5 / @10 | 6/10 (60%) |
| Latency p50 / p95 | 3.04 ms / 3966.51 ms |

**Five golden failures, and only four of them are the corpus.** `q04`, `q07`,
`q08` and `q10` ask for `lock_in`, `statement_guide`, `tax_benefit` and
`direct_vs_regular`, none of which has a single chunk in the corpus, so recall is
capped at 6/10 by the *data*.

The fifth is `q02`, and it is the interesting one. It asks for the exit load on
HDFC Equity Fund - Direct Growth. The corpus **does** contain "Exit load: 1% if
redeemed within 1 year", and retrieval ranks that chunk **first**. The model still
answers:

> I do not have the exit load for HDFC Equity Fund - Direct Growth in my sources.
> **Source:** https://groww.in/… **Last updated from sources: 27 Sep 2026`

That used to score as a full pass. It is a non-answer wearing a citation: the
`Source:` and `Last updated` lines are rendered from the top chunk's *metadata*,
so they appear whether or not the chunk contains the answer. Route, citation,
freshness, sentence count and `field_type` all passed. The eval now has a
`cited_answer_is_not_a_refusal` check that catches it, which is why this run reads
**5/10 and 20/25 rather than the 6/10 and 21/25 it reported before that check
existed.** The lower number is the correct one.

Also worth reading closely:

- **Both refusals are successes, and there are 0 outages.** `q08` and `q10` clear
  no chunk above the similarity floor. `results.json` counts a refusal separately
  from a provider outage (`degraded` vs `service_failure`).
- **Latency p95 is ~4 s** while p50 is ~3 ms. The p50 rows are the
  keyword-routed refusals, which never call the model. The p95 is a live
  Llama-class generation. This is a real number for a hosted API, not a
  tuned-away one.

**Re-run `make eval` to reproduce.** These are one run's numbers, not a claim
about every future run.

---

## SCOPE

AMC: **HDFC Mutual Fund**. Five schemes, all Direct Growth plans:

| Scheme | Category | Source page |
|---|---|---|
| HDFC Large Cap Fund - Direct Growth | Large cap | https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth |
| HDFC Equity Fund - Direct Growth (Flexi Cap) | Flexi cap | https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth |
| HDFC ELSS Tax Saver Fund - Direct Growth | ELSS | https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth |
| HDFC Small Cap Fund - Direct Growth | Small cap | https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth |
| HDFC Balanced Advantage Fund - Direct Growth | Hybrid | https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth |

---

## SETUP

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# edit .env: set LLM_PROVIDER, LLM_MODEL and that provider's key
make ingest
make run
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env: set LLM_PROVIDER, LLM_MODEL and that provider's key
make ingest
make run
```

Without `make` (it is not installed by default on Windows):

```powershell
.\.venv\Scripts\python.exe -m ingest.run_ingest
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The embedding model `sentence-transformers/all-MiniLM-L6-v2` downloads roughly
90 MB on the first run only, and is offline from then on.

Ask questions from the terminal instead of the browser:

```powershell
.\.venv\Scripts\python.exe -m rag.pipeline -i                              # real answers, needs a key
.\.venv\Scripts\python.exe -m eval.inspect_query -i                        # retrieval only, no key needed
```

---

## ARCHITECTURE

Two flows over one corpus and one vector store. Flow A runs offline; Flow B runs
per question.

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
│  contract      PII/ADVICE/       top-k +       grounded      ≤3 sent ·     │
│  + 3 ex. Qs    PERF/OOS gate      scheme +      context-      1 link ·      │
│  + disclaimer  field_type        only          freshness                  │
│                boost                                         boost         │
└──────────────────────────────────────────────────────────────────────────────┘
```

The load-bearing decision: **the LLM contributes only the answer prose.** The
`Source:` line and the `Last updated from sources:` line are assembled in Python
from the winning chunk's metadata, so the citation and freshness guarantees
cannot be violated by the model.

---

## CHUNKING

Strategy `auto`, with a label/value pass that runs first and unconditionally on
every document. Real settings from `config.yaml`: `target_tokens=600`,
`min_tokens=200`, `max_tokens=800`, `overlap_pct=12`.

The reason a fixed-size split was rejected: these pages clean down to one token
per line, so a size-based split emits `1.03%` as its own chunk tagged `other` —
retrievable, plausible-looking, and attributable to the wrong subject. The
governing rule became **a value is never separated from its label.** The full
The full reasoning, including the guards against confidently-wrong
facts, is reproduced below from [`chunking_rationale.md`](chunking_rationale.md):

Why `ingest/chunk.py` chunks the way it does. Required by
`docs/implementation.md` Phase 3; the decisions below were all forced by looking
at the real pages, and several of them contradict the first draft of
`docs/architecture.md` §5.4.

### The decision that mattered most

A fixed-size character split was rejected from the start, because it destroys the
two units every fact lookup depends on: a FAQ question/answer pair, and a table
row. That reasoning turned out to be **understated**. On the real corpus a
fixed-size split does not merely degrade retrieval, it produces chunks that are
*actively wrong*, and the failure mode is silent.

The published scheme pages clean down to one token per line:

```
Expense ratio
1.03%
```

Split on size, `1.03%` becomes a chunk of its own, tagged `other`. It is
retrievable, it looks like a legitimate result, and it will be quoted with a
correct-looking citation attached to a value whose subject is not in the chunk at
all. Worse, it usually lands in a *large* chunk next to an exit load, so the
generator sees `... Exit load: 1% if redeemed within 1 year ... 1.03% ...` and
can attribute either number to the other.

So the rule became: **a value is never allowed to be separated from its label.**
That single invariant drove most of what follows.

### The four document shapes

`profile()` classifies each document, but the shape only decides how the
*leftovers* are split. See "Hybrid pages" below for why that is not enough on
its own.

| Shape | Splitter | Why |
|---|---|---|
| `FAQ` | `_split_faq` | A question and its answer are one unit; a pair is never cut, even if it runs past `max_tokens`. |
| `TABLE` | `_table_parts` | Each row is prefixed with the header so a cell is self-describing. |
| `LABEL_VALUE` | `_split_label_value` | Re-pairs `Label` / `value` line pairs into `Label: value`. |
| `NARRATIVE` | `_split_narrative` | Heading-anchored, then sentence, then word. |

### Hybrid pages, and why the shape switch is not the answer

Every scheme page is a hybrid: a key-value facts block, then a large holdings
table, then prose, then a "what would this have grown into" calculator. No single
shape describes one.

The first implementation picked a shape and split the whole document with that
splitter. On the real corpus this produced **16 chunks for 66,000 characters**,
and the entire facts block ended up inside one multi-kilobyte chunk tagged
`exit_load` — because the word "exit load" happened to appear somewhere inside
it. `Expense ratio: 1.03%` was present in the corpus and completely unretrievable
as a fact.

So the label/value pass now runs **first and unconditionally**, on every
document, and only the lines it does not claim are handed to the shape-appropriate
splitter. Each fact is emitted as its own atomic chunk. Corpus size went from 16
to ~1000 chunks, and `expense_ratio`, `benchmark`, `min_sip`, `exit_load` and
`nav` became individually retrievable with correct tags.

### Guards the splitter needs on real data

Each of these is a bug that produced a *confidently wrong* fact, not a degraded
ranking.

**A bare horizon is not a lock-in.** The pages label an axis `3 years` directly
above `₹1,80,000` in the growth calculator. Paired naively, that becomes
  `3 years: ₹1,80,000`, tagged `lock_in` — and it appears on *all five* scheme
  pages, because Groww shows the tax-saver blurb everywhere. A large-cap lock-in
  question would then be answered with a growth projection, citing the large-cap
  page. A bare horizon is now only a value when a fact label owns it
  (`Lock-in period` / `3 years`), and the `lock_in` rule requires lock-in
  vocabulary before a bare duration counts at all. No chunk in the current corpus
  contains `₹1,80,000`.

**Returns are dropped, not just deprioritised.** The PRD forbids computing or
quoting returns. Leaving performance tables in the corpus leaves a generator free
to quote them, so units with two or more percentages plus a performance hint are
dropped at ingest. The test is deliberately conservative so that
`Expense ratio: 1.03%` and `Lock-in period: 3 years` survive.

**Other funds' facts on this page are removed.** Each page carries an "Also
manages these schemes" link list and benchmark tables covering dozens of other
HDFC funds. Left alone, a question about one scheme can be grounded in a
neighbouring page's number and answered with a correct-looking citation to the
wrong fund (PRD A14). Units whose subject is a *named fund* that is not the
document's own scheme are dropped. This required adding a `page_title` to
`SchemeSpec`: the registry calls the flexi cap fund "HDFC Equity Fund" but the
page titles it "HDFC Flexi Cap", so matching on `display_name` alone would have
let every Flexi Cap fact through as contamination.

**Chunks never merge across a section boundary, and never merge two different
facts.** `_merge_and_overlap` used to carry the first section's heading forward
and merge whatever followed. The scheme "About" paragraph was therefore absorbed
into the preceding "Return calculator" section and then dropped as performance —
taking the riskometer with it. Merging is now confined to a single section, and
two sentences stating different field types stay apart so each chunk carries one
meaningful `field_type` for the Phase 5 retrieval boost.

### The size guarantee has to be real

`_group_paragraphs` hard-split oversized paragraphs at a hardcoded 800 tokens
while `max_tokens` was 140. The tests missed it because the fixtures contained
blank lines, so paragraphs formed. **The cleaned corpus contains no blank lines at
all**, so an entire section arrives as one paragraph, and holdings rows carry no
sentence punctuation to break on. Such a run could not be split at all and landed
in the corpus as a single enormous chunk.

The hard split therefore escalates: sentence boundary, then word boundary. The
word-level pass is what makes the guarantee hold, and it is covered by
`test_unpunctuated_runs_are_still_size_bounded`.

### Tagging

`tag_field_type` picks the **earliest** matching rule, breaking ties toward the
longer (more specific) match. Earliest-wins matters: *"The scheme is rated Very
High risk. Minimum SIP Investment is set to ₹500."* is a riskometer fact, and
rule-order would have called it `min_sip`. Longer-on-ties keeps *"direct plan
versus regular"* from being outvoted by a bare *"direct plan"*.

`direct_vs_regular` deliberately requires a comparison rather than a bare
mention of "direct plan", which appears in nearly every chunk of this corpus since
every scheme is a Direct Growth plan. Matching it would retag the whole document
and make the Phase 5 boost useless.

### Known limits

- **Holdings and portfolio rows dominate the corpus.** ~91% of chunks (959 of
  1055) are `other`, and 488 of those come from the balanced advantage fund's
  single 33,000-character page of debenture and equity tables (which supplies
  511 of the 1055 chunks in total). Holdings are out
  of scope for the assistant, so they are pure retrieval noise. They are retained
  because detecting them reliably is harder than the Phase 5 similarity floor
  that will filter them, but this is the first thing to revisit if retrieval
  quality suffers.
- **Riskometer is present on 3 of 5 pages.** The flexi cap and ELSS pages do not
  state it in the cleaned text. This is a source gap, not a chunking gap — see
  the primary-source note in `docs/PRD.md`.
- **The ELSS lock-in is not explicitly labelled anywhere** in the current corpus.
  `lock_in` is therefore empty by design rather than by accident; the old
  `3 years: ₹1,80,000` entry was the growth calculator and was removed as
  contamination. Answering a lock-in question needs a real KIM/SID/factsheet.

### Phase 5 retrieval operating point (frozen)

Measured on `eval/golden_set.jsonl` at `corpus=1055` chunks with
`python -m eval.recall`:

```
over_fetch:         10   (was 2)
similarity_floor: 0.35   (unchanged)
field_type_boost: 0.35   (was 0.05)
recall@1  5/10
recall@3  6/10
recall@5  6/10
```

**6/10 is the ceiling for this corpus, not a retrieval shortfall.** Four of the
ten golden queries ask for facts that were never ingested, so no threshold
setting can find them:

| query | needs | chunks in corpus |
| --- | --- | --- |
| q04 | `lock_in` | 0 |
| q07 | `statement_guide` | 0 |
| q08 | `tax_benefit` | 0 |
| q10 | `direct_vs_regular` | 0 |

All four sit in the six disabled primary-source rows of `ingest/sources.csv`.
Enabling them with real URLs is the only way past 6/10.

**Why the boost had to become 0.35.** The holdings and `scheme_type` noise
described above embeds close to *any* query naming a fund, because those chunks
are literally long lists of other funds' names. Before tuning, the correct fact
chunk ranked 9th-13th, outside the old 7-candidate pool:

| query | fact chunk | rank | similarity | top-ranked chunk | similarity |
| --- | --- | --- | --- | --- | --- |
| q01 | `Expense ratio: 1.03%` | 9 | 0.484 | `scheme_type` noise | 0.755 |
| q02 | `Exit load: ... 1% if redeemed within 1 year` | 13 | 0.401 | `scheme_type` noise | 0.773 |
| q05 | `Fund benchmark: NIFTY 50 Hybrid ...` | 13 | 0.495 | `scheme_type` noise | 0.837 |

A 0.05 additive boost cannot close a 0.27 gap, so `over_fetch` was raised to
admit the fact chunk and the boost raised until it could win. A boost as large
as the floor is not a subtle number; it is a debt owed to the corpus noise.
**If the noise is removed at source, both values should fall back to small ones
and this section should be re-measured rather than re-tuned.**

The floor is insensitive between 0.30 and 0.40 at this boost, so 0.35 is kept
as the documented value instead of being over-fitted to the noise level.

## TUNING

From `config.yaml`, measured on `eval/golden_set.jsonl` with `python -m eval.recall`:

| Parameter | Value | Why |
|---|---|---|
| `retrieval.top_k` | 10 | This is the LLM context window — chunks shown to the model. See below. |
| `retrieval.over_fetch` | 10 | Was 2. The correct fact chunk ranked 9th–13th, outside a 7-candidate pool. |
| `retrieval.similarity_floor` | 0.35 | Insensitive between 0.30 and 0.40 at this boost, so it is not over-fitted. |
| `retrieval.field_type_boost` | 0.35 | Was 0.05. A 0.05 boost cannot close a 0.27 similarity gap. |
| `retrieval.field_misfit_penalty` | 0.15 | Demotes chunks about *other schemes* when a field intent was detected. |
| `retrieval.scheme_filter` | `strict` | Cross-scheme contamination is the highest-severity risk (A14). |

A boost as large as the floor is not a subtle number — it is debt owed to corpus
noise. Holdings and `scheme_type` chunks embed close to any query naming a fund,
because they are long lists of other funds' names. If that noise is removed at
source, both values should fall back to small ones and this table should be
re-measured rather than re-tuned.

**Why the context window is 10.** `build_context` renders *every* chunk in the
trace, so `top_k` is not just a retrieval depth — it is literally how much corpus
the model reads. I measured 5 against 10 on the same corpus:

| | `top_k=5` | `top_k=10` |
|---|---|---|
| Overall | 21/25 | 21/25 |
| Golden recall@1 | 6/10 | 6/10 |
| Golden generation checks | 36/39 | **33/39** |
| Rows that refused instead of answering | 1 | 2 |

Nothing improved, and one metric went down. The interesting part is *why*, and it
is not a regression.

`q10` asks for the direct-vs-regular expense ratio. The corpus has **zero**
`direct_vs_regular` chunks. At `top_k=5` the system still answered — with a
non-answer that carried a source line:

> I do not have the expense ratio details … in my sources.
> **Source:** https://groww.in/… **Last updated from sources: 27 Sep 2026**

It cleared every mechanical check. The citation and freshness lines are rendered
from the top retrieved chunk's *metadata*, and that chunk was a holdings table
which does not contain the answer. So a refusal was dressed as an evidenced
answer, and the eval scored it as one.

At `top_k=10` the model can see that the whole neighbourhood is about other
things, and it refuses plainly instead. That is the contract working, and it is
the behaviour I want.

The honest cost of 10: input tokens per generation roughly double, on a hosted
API. Latency was a wash here. Set `retrieval.top_k: 5` in `config.yaml` to A/B
it; recall will not change, because 6/10 is the corpus ceiling.

**One thing the width sweep did *not* fix.** Sweeping `top_k` over 1, 2, 3, 5 and
10 showed that `q02` refuses at *every* width, including the most favourable case
of all — a three-chunk window whose first entry is the exit load itself:

| `top_k` | First chunk in the window | `q02` answer |
|---|---|---|
| 1 | `scheme_type` | refusal |
| 2 | `scheme_type` | refusal |
| 3 | **`exit_load`** | refusal |
| 5 | **`exit_load`** | refusal |
| 10 | **`exit_load`** | refusal |

So the wider window is not the cause of that failure, and no width fixes it. It
is the model declining to use a chunk that is explicitly labelled and does
contain the value. Narrowing the window would have hidden the problem without
solving it, which is why the honest 5/10 above is worth more than the 6/10 it
replaced.

**Why the misfit penalty exists.** Boosting can only promote the right chunk so
far. On `q02` ("exit load on HDFC Equity Fund - Direct Growth") a `scheme_type`
chunk listing sibling funds still won: 0.773 raw against 0.751 for the correct
`exit_load` chunk, a 0.022 loss that would have answered a field question with a
fund list. Closing a 0.37 raw gap with a bigger boost would distort genuine
matches, so the fix is the other side of the ledger: when a field intent *is*
detected, chunks tagged `scheme_type` are off-topic by construction and are
demoted instead. Golden top-1 went 5/10 → 6/10, and — the part that makes this
trustworthy — it is **flat at 6/10 for every penalty from 0.05 to 0.40**, with no
other query changing state. 0.15 is a plateau midpoint, not a fitted value, and
6/10 is the corpus ceiling rather than the ceiling of this parameter. Set
`field_misfit_penalty: 0.0` to disable the behaviour and A/B it.

---

## TECH STACK

| Component | Choice |
|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`, 384-d, L2-normalised |
| Vector store | ChromaDB, persistent, local, cosine (`mf_faq_chroma`) |
| UI | Streamlit 1.64 |
| LLM | Any OpenAI-compatible endpoint. Shipped config: Groq serving `qwen/qwen3.8-27b`, `temperature=0.0`. Every run records the actual provider and model in `eval/results.json`. |
| Tests | pytest, 557 tests |

---

## GUARDRAILS

Routing happens **before** retrieval, so a refused question never reaches the
vector store and never reaches the model.

| Route | Triggered by | Behaviour |
|---|---|---|
| **PII** | PAN, Aadhaar (Verhoeff-checked), account/folio, card, OTP, email, phone | Rejected before retrieval. Neutral message, no citation. The input is never logged or echoed back. |
| **ADVICE** | "should I", "which is best", "worth buying", "suitable for me", allocation language | Facts-only refusal with the SEBI investor-education link. |
| **PERFORMANCE** | returns, CAGR, "best performing", ranking, growth claims | Redirect to the official factsheet. No figures, ever. |
| **OUT_OF_SCOPE** | anything not answerable from the five pages (e.g. today's NAV) | "Not in my sources" with no invented answer. |
| **FACTUAL** | everything else | Retrieved, generated from context only, then validated. |

PII detection is deterministic and keyword-gated where a bare number is
ambiguous, so `Expense ratio: 1.03%` is never mistaken for an account number. A
16-digit card number is caught even though no other class would match it.

The query is never written to disk. `logs/app.log` receives a fixed set of keys
(`ts, stage, route, scheme, n_chunks, top_score, validators_passed, latency_ms,
outcome, degraded, matched_classes, source_id`) with PAN, email, phone and long
digit runs redacted.

---

## DELIVERABLES

| ID | File | What it is |
|---|---|---|
| D1 | — | Demo video (≤ 3 min, UI only) |
| D2 | [`source_list.csv`](source_list.csv) | Every source document with URL, retrieval date, SHA-256 and status |
| D3 | this `README.md` | Scope, setup, architecture, chunking, tuning, limits |
| D4 | [`sample_qa.md`](sample_qa.md) | Real transcribed answers covering all four behaviours |
| D5 | [`disclaimer.md`](disclaimer.md) | The disclaimer, generated from the same constant the UI renders |

---

## KNOWN LIMITS

1. **Point-in-time snapshot.** Expense ratios and exit loads change. Every answer
   carries a freshness date; the corpus is a snapshot, not a live feed.
2. **Four of ten golden queries are unanswerable from this corpus**, which caps
   recall at 6/10 no matter how the retrieval is tuned. `lock_in`,
   `statement_guide`, `tax_benefit` and `direct_vs_regular` have **zero** chunks
   each (golden rows `q04`, `q07`, `q08`, `q10`). The 6/10 measured at k≥3 is
   therefore the ceiling, not a tuning shortfall. Those facts live in the six
   disabled primary-source rows of `ingest/sources.csv`.
3. **The ELSS lock-in is not labelled anywhere in the corpus.** Answering it
   needs a real KIM/SID/factsheet, not a tuning change.
4. **MiniLM-L6-v2 may blur direct vs regular.** A small model on a corpus where
   every scheme is a Direct Growth plan. Mitigated by strict scheme filtering and
   `field_type` boosting, not eliminated.
5. **Source authority.** The seed URLs are Groww, a registered distributor
   platform, not the AMC. All five ingested documents are Groww scheme pages.
   Primary AMC/SEBI/AMFI documents are listed in `ingest/sources.csv` and are
   preferred where available, but six of them are currently disabled for want of a
   real URL. Flagged for mentor confirmation.
6. **~91% of chunks are noise.** 959 of 1055 chunks are `other`, and 488 of those
   come from one 33,000-character holdings table (the balanced advantage page,
   which supplies 511 of the 1055 chunks in total). Holdings are out of
   scope, so this is pure retrieval noise and the reason the boost is as large as it is.
7. **Return figures are still in the corpus; the guard is at query time, not at
   ingest.** 24 of 1055 chunks contain return-like text and at least one carries an
   actual figure. Nothing strips them during ingestion. What prevents them being
   quoted is the `PERFORMANCE` route refusing before retrieval plus a prompt
   instruction — a working guardrail, but not a data guarantee. A removal pass at
   ingest would be strictly stronger, and is the honest fix if routing or prompts
   ever drift.
8. **Narrow scope by design.** Five schemes, one AMC. Adding schemes is a config
   and manifest change.
9. **The model can decline an answer that is sitting in the window.** Golden `q02`
   asks for the exit load on HDFC Equity Fund - Direct Growth. The corpus
   contains it, ranked first, and the model still replies "I do not have the
   exit load … in my sources" with a rendered citation attached. Sweeping
   `top_k` over 1, 2, 3, 5 and 10 reproduces it at every width, including a
   three-chunk window whose first entry is the answer itself, so it is a
   generation failure rather than a retrieval or context-budget one. The
   `cited_answer_is_not_a_refusal` check exists to keep it visible; fixing it
   means changing the prompt or the model, and it is the single most valuable
   open item here.

---

## REPRODUCIBILITY

```bash
make setup && make ingest && make eval && make run
```

Or without `make`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ingest.run_ingest
.\.venv\Scripts\python.exe -m eval.run_eval --source-list --sample-qa
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Ingestion is idempotent at the store: chunk ids are content-derived, so re-running
ingest rewrites the same ids and adds no duplicates
(`tests/test_store_idempotency.py`). `make eval` regenerates
`eval/results.json`, `docs/source_list.csv` and `docs/sample_qa.md`, so no number
in this repository is maintained by hand.
