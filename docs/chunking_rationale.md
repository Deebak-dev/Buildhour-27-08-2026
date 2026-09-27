# Chunking Rationale

Why `ingest/chunk.py` chunks the way it does. Required by
`docs/implementation.md` Phase 3; the decisions below were all forced by looking
at the real pages, and several of them contradict the first draft of
`docs/architecture.md` §5.4.

## The decision that mattered most

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

## The four document shapes

`profile()` classifies each document, but the shape only decides how the
*leftovers* are split. See "Hybrid pages" below for why that is not enough on
its own.

| Shape | Splitter | Why |
|---|---|---|
| `FAQ` | `_split_faq` | A question and its answer are one unit; a pair is never cut, even if it runs past `max_tokens`. |
| `TABLE` | `_table_parts` | Each row is prefixed with the header so a cell is self-describing. |
| `LABEL_VALUE` | `_split_label_value` | Re-pairs `Label` / `value` line pairs into `Label: value`. |
| `NARRATIVE` | `_split_narrative` | Heading-anchored, then sentence, then word. |

## Hybrid pages, and why the shape switch is not the answer

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

## Guards the splitter needs on real data

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

## The size guarantee has to be real

`_group_paragraphs` hard-split oversized paragraphs at a hardcoded 800 tokens
while `max_tokens` was 140. The tests missed it because the fixtures contained
blank lines, so paragraphs formed. **The cleaned corpus contains no blank lines at
all**, so an entire section arrives as one paragraph, and holdings rows carry no
sentence punctuation to break on. Such a run could not be split at all and landed
in the corpus as a single enormous chunk.

The hard split therefore escalates: sentence boundary, then word boundary. The
word-level pass is what makes the guarantee hold, and it is covered by
`test_unpunctuated_runs_are_still_size_bounded`.

## Tagging

`tag_field_type` picks the **earliest** matching rule, breaking ties toward the
longer (more specific) match. Earliest-wins matters: *"The scheme is rated Very
High risk. Minimum SIP Investment is set to ₹500."* is a riskometer fact, and
rule-order would have called it `min_sip`. Longer-on-ties keeps *"direct plan
versus regular"* from being outvoted by a bare *"direct plan"*.

`direct_vs_regular` deliberately requires a comparison rather than a bare
mention of "direct plan", which appears in nearly every chunk of this corpus since
every scheme is a Direct Growth plan. Matching it would retag the whole document
and make the Phase 5 boost useless.

## Known limits

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

## Phase 5 retrieval operating point (frozen)

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