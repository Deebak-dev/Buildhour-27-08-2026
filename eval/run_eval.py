"""Evaluation harness (Phase 9, docs/implementation.md).

Runs the two evaluation suites and writes ``eval/results.json``:

* **golden** - the 10 factual queries in ``eval/golden_set.jsonl``. Each answer is
  checked against the §7 response contract, and recall@k is computed from the
  chunks the pipeline actually retrieved.
* **adversarial** - the 15 routing and PII queries in
  ``eval/adversarial_set.jsonl``.

Every number in the output and in the deliverables comes from this run. Nothing
here is hand-written, and a failing check is reported as a failure rather than
explained away: a pass rate is only worth reading if it can go down.

Usage::

    python -m eval.run_eval                  # both suites
    python -m eval.run_eval --golden --show  # golden set, full answers
    python -m eval.run_eval --k 1 3 5 10     # recall at these cutoffs
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from rag.config import PROJECT_ROOT
from rag.schemas import Route

GOLDEN_PATH = PROJECT_ROOT / "eval" / "golden_set.jsonl"
ADVERSARIAL_PATH = PROJECT_ROOT / "eval" / "adversarial_set.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "eval" / "results.json"

#: Lines the contract adds below the prose. They are not sentences of the answer,
#: so they are stripped before the ≤ 3-sentence check (criterion A4).
CONTRACT_LINES = ("Source:", "Last updated from sources:")

DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10)

#: Phrases that mark prose as a refusal rather than an answer. Matched as a
#: family, not against the canonical strings, because the model writes its own
#: refusals too. All are lower-case; the prose is lower-cased before matching.
REFUSAL_PHRASES = (
    "do not have",
    "don't have",
    "dont have",
    "not in my sources",
    "isn't in my sources",
    "is not in my sources",
    "could not find",
    "cannot find",
    "no information on",
    "no information about",
)

#: Literals that must never survive into an answer to a PII query.
_SECRET_PATTERNS = (
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
    re.compile(r"(?<![\d])\d{4,}(?![\d])"),
)


# ---------------------------------------------------------------------------
# Row loading
# ---------------------------------------------------------------------------
def load_rows(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL suite file, skipping ``#`` comment headers and blank lines."""
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        rows.append(json.loads(stripped))
    return rows


def prose_of(text: str) -> str:
    """The answer without the contract's ``Source:``/``Last updated`` lines."""
    kept = [
        line
        for line in (text or "").splitlines()
        if not line.strip().startswith(CONTRACT_LINES)
    ]
    return "\n".join(kept).strip()


def sentence_count(text: str) -> int:
    """Sentences in the prose only. Decimals do not split (``1.03%`` is one)."""
    prose = prose_of(text)
    if not prose:
        return 0
    return len([s for s in re.split(r"(?<=[.!?])\s+", prose) if s.strip()])


def source_urls(text: str) -> list[str]:
    """Every URL on a ``Source:`` line."""
    return [
        line.split("Source:", 1)[1].strip()
        for line in (text or "").splitlines()
        if line.strip().startswith("Source:")
    ]


def secrets_in(query: str) -> list[str]:
    """Sensitive-looking literals in a query, for the PII no-echo check."""
    found: list[str] = []
    for pattern in _SECRET_PATTERNS:
        found.extend(m.group(0) for m in pattern.finditer(query))
    return found


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""
    #: True when the check can only pass if the model actually produced prose.
    #: Citation, freshness, and sentence count all depend on generation; route,
    #: field_type, and scheme purity do not. Splitting them stops an API-key
    #: outage from being reported as though retrieval had failed.
    needs_generation: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "needs_generation": self.needs_generation,
        }


@dataclass
class RowResult:
    suite: str
    row_id: str
    query: str
    passed: bool
    checks: list[Check] = field(default_factory=list)
    latency_ms: float = 0.0
    answer: str = ""
    route: str = ""
    degraded: bool = False
    #: Narrower than ``degraded``: True only when generation could not run (no
    #: client, provider error, rate limit). A "not in my sources" refusal is
    #: also degraded, but it is the system behaving correctly, and the report
    #: must not present it as an outage.
    service_failure: bool = False
    hits_at: dict[int, bool] = field(default_factory=dict)

    @property
    def failures(self) -> list[str]:
        return [
            f"{c.name}: {c.detail}" for c in self.checks if not c.passed
        ]


# ---------------------------------------------------------------------------
# Golden suite
# ---------------------------------------------------------------------------
def is_refusal_prose(text: str) -> bool:
    """Whether the answer's *own* prose declines to answer.

    The canonical strings live in :data:`rag.prompts.REFUSALS`, but the model
    also writes refusals in its own words ("I do not have the exit load for
    HDFC Equity Fund - Direct Growth in my sources"), so the family has to be
    matched rather than the exact text. Matching the family is deliberate: a
    refusal is defined by what it declines to state, not by its phrasing.
    """
    prose = prose_of(text).lower()
    return any(phrase in prose for phrase in REFUSAL_PHRASES)


def check_golden_row(
    row: dict[str, Any], payload: Any, ks: Sequence[int] = DEFAULT_KS
) -> list[Check]:
    """The contract checks for one golden row.

    A check whose expectation is ``None`` is not applicable and is skipped rather
    than counted as a pass, so an out-of-corpus row is not silently credited.
    """
    text = payload.text or ""
    trace = getattr(payload, "trace", None)
    chunks = list(getattr(trace, "chunks", None) or [])

    urls = source_urls(text)
    expected_url = row.get("expect_source_url")
    expected_scheme = row.get("scheme")
    expected_field = row.get("expect_field_type")

    # A refusal carrying a rendered citation is the one failure mode that every
    # other check here rewards: the citation is built from the top chunk's
    # metadata whether or not that chunk contains the answer, so a non-answer
    # can pass route, citation, freshness and sentence-count while telling the
    # user nothing. Real queries hit this - golden q02 says "I do not have the
    # exit load ..." with the exit load ranked first in the window.
    cited_refusal = bool(urls) and is_refusal_prose(text)

    checks = [
        Check(
            "route_is_factual",
            payload.route is Route.FACTUAL,
            f"got {getattr(payload.route, 'value', payload.route)}",
        ),
        Check(
            "cited_answer_is_not_a_refusal",
            not cited_refusal,
            "prose refuses to answer but a Source: line was rendered, so a "
            "non-answer reads as an evidenced one"
            if cited_refusal
            else "",
            needs_generation=True,
        ),
        Check(
            "one_source_line",
            len(urls) == 1,
            f"found {len(urls)} 'Source:' line(s)",
            needs_generation=True,
        ),
        Check(
            "has_freshness_line",
            "Last updated from sources:" in text,
            "freshness line missing",
            needs_generation=True,
        ),
        Check(
            "max_3_sentences",
            sentence_count(text) <= 3,
            f"{sentence_count(text)} sentences",
            needs_generation=True,
        ),
    ]

    if expected_url is not None:
        checks.append(
            Check(
                "citation_matches_expected",
                urls == [expected_url],
                f"got {urls or ['<none>']}, want ['{expected_url}']",
                needs_generation=True,
            )
        )

    if expected_field is not None:
        top_field = chunks[0].chunk.field_type if chunks else None
        checks.append(
            Check(
                "top_chunk_field_type",
                top_field == expected_field,
                f"top chunk is {top_field!r}, want {expected_field!r}",
            )
        )

    if expected_scheme is not None:
        others = sorted({c.chunk.scheme for c in chunks} - {expected_scheme})
        checks.append(
            Check(
                "scheme_purity",
                not others,
                f"cross-scheme chunks: {others}" if others else "all chunks in-scheme",
            )
        )

    return checks


def recall_hits(payload: Any, expected_field: str | None, ks: Iterable[int]) -> dict[int, bool]:
    """Whether a chunk of the expected ``field_type`` appears in the top k."""
    if expected_field is None:
        return {}
    chunks = list(getattr(getattr(payload, "trace", None), "chunks", None) or [])
    return {
        int(k): any(c.chunk.field_type == expected_field for c in chunks[: int(k)])
        for k in ks
    }


# ---------------------------------------------------------------------------
# Adversarial suite
# ---------------------------------------------------------------------------
def check_adversarial_row(row: dict[str, Any], payload: Any) -> list[Check]:
    """Route must match, and a PII answer must not echo the secret back."""
    expected = row.get("expect_route")
    got = getattr(payload.route, "value", payload.route)
    checks = [
        Check("route_matches", got == expected, f"got {got}, want {expected}")
    ]

    if expected == Route.PII.value:
        text = payload.text or ""
        query = row.get("query", "")
        echoed = [s for s in secrets_in(query) if s and s in text]
        checks.append(
            Check(
                "pii_not_echoed",
                not echoed and query.strip() not in text,
                f"echoed {echoed}" if echoed else "no secret in the answer",
            )
        )
    return checks


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------
def run_suite(
    suite: str,
    rows: Sequence[dict[str, Any]],
    ask_fn: Callable[[str], Any],
    *,
    ks: Sequence[int] = DEFAULT_KS,
    checker: Callable[..., list[Check]],
) -> list[RowResult]:
    """Ask every row and collect its checks. A raising row is a failure, not a crash."""
    results: list[RowResult] = []
    for row in rows:
        query = row.get("query", "")
        started = time.perf_counter()
        try:
            payload = ask_fn(query)
        except Exception as exc:  # noqa: BLE001 - a crash is a failed row
            results.append(
                RowResult(
                    suite=suite,
                    row_id=row.get("id", "?"),
                    query=query,
                    passed=False,
                    checks=[
                        Check("ask_raised", False, f"{type(exc).__name__}: {exc}")
                    ],
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                )
            )
            continue
        latency = (time.perf_counter() - started) * 1000.0

        checks = (
            checker(row, payload, ks)
            if suite == "golden"
            else checker(row, payload)
        )
        results.append(
            RowResult(
                suite=suite,
                row_id=row.get("id", "?"),
                query=query,
                passed=all(c.passed for c in checks),
                checks=checks,
                latency_ms=latency,
                answer=payload.text or "",
                route=getattr(payload.route, "value", str(payload.route)),
                degraded=bool(getattr(payload, "degraded", False)),
                service_failure=bool(getattr(payload, "service_failure", False)),
                hits_at=recall_hits(payload, row.get("expect_field_type"), ks),
            )
        )
    return results


def _percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile; 0.0 for an empty sample."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(pct / 100.0 * len(ordered)) - 1))
    return round(ordered[index], 2)


def summarise(
    results: Sequence[RowResult], ks: Sequence[int] = DEFAULT_KS
) -> dict[str, Any]:
    """Pass rates, recall@k and latency for one suite or for everything."""
    by_suite: dict[str, list[RowResult]] = {}
    for result in results:
        by_suite.setdefault(result.suite, []).append(result)

    suites: dict[str, Any] = {}
    for name, rows in sorted(by_suite.items()):
        passed = sum(1 for r in rows if r.passed)
        recall: dict[str, Any] = {}
        if name == "golden":
            for k in ks:
                applicable = [r for r in rows if r.hits_at.get(int(k)) is not None]
                if applicable:
                    hits = sum(1 for r in applicable if r.hits_at[int(k)])
                    recall[f"recall@{k}"] = round(hits / len(applicable), 4)
                    recall[f"recall@{k}_hits"] = f"{hits}/{len(applicable)}"

        # Check tallies split by whether generation was required. The
        # generation-independent column is the one that stays meaningful during
        # an API outage; the other is reported so nothing is hidden.
        tallies: dict[str, Any] = {}
        for label, wants in (
            ("retrieval_checks", False),
            ("generation_checks", True),
        ):
            subset = [c for r in rows for c in r.checks if c.needs_generation is wants]
            if subset:
                ok = sum(1 for c in subset if c.passed)
                tallies[label] = {
                    "passed": ok,
                    "total": len(subset),
                    "pass_rate": round(ok / len(subset), 4),
                }

        suites[name] = {
            "n": len(rows),
            "passed": passed,
            "pass_rate": round(passed / len(rows), 4) if rows else 0.0,
            "degraded": sum(1 for r in rows if r.degraded),
            "service_failures": sum(1 for r in rows if r.service_failure),
            **tallies,
            **recall,
        }

    latencies = [r.latency_ms for r in results]
    total = len(results)
    return {
        "suites": suites,
        "overall_pass_rate": (
            round(sum(1 for r in results if r.passed) / total, 4) if total else 0.0
        ),
        "overall_passed": sum(1 for r in results if r.passed),
        "overall_n": total,
        "latency_ms": {
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
            "mean": round(statistics.fmean(latencies), 2) if latencies else 0.0,
        },
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _tick(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def print_table(results: Sequence[RowResult], show: bool = False) -> None:
    """A plain-text, ANSI-free per-check table."""
    for result in results:
        head = f"{result.row_id}  {_tick(result.passed)}  [{result.route}]"
        print(f"\n{head}")
        print(f"  query: {result.query}")
        for check in result.checks:
            line = f"  {_tick(check.passed)}  {check.name}"
            if check.detail and not check.passed:
                line += f" - {check.detail}"
            print(line)
        if result.hits_at:
            hits = ", ".join(
                f"@{k}={'y' if v else 'n'}" for k, v in sorted(result.hits_at.items())
            )
            print(f"  recall: {hits}")
        if show and result.answer:
            print("  --- answer ---")
            for line in result.answer.splitlines():
                print(f"  {line}")


def build_report(
    results: Sequence[RowResult], ks: Sequence[int] = DEFAULT_KS
) -> dict[str, Any]:
    """The ``results.json`` document, with every failure and its reason."""
    from rag.config import load_config

    summary = summarise(results, ks)
    degraded = sum(1 for r in results if r.degraded)
    outages = sum(1 for r in results if r.service_failure)
    notes: list[str] = []
    if degraded:
        notes.append(
            f"{degraded} of {len(results)} rows returned a non-answer "
            "(degraded). Of those, "
            f"{outages} were genuine generation failures (no API key, provider "
            "error, or rate limit) and "
            f"{degraded - outages} were the contract working correctly: a gate "
            "downgrade or a 'not in my sources' refusal. Contract checks on "
            "either kind of row measure the refusal path, not answer quality."
        )
    if degraded and outages == 0:
        notes.append(
            "No row failed because of a provider outage. Any non-answer here is "
            "a deliberate refusal."
        )
    try:
        cfg = load_config().generation
        provider, model = cfg.provider, cfg.model
    except Exception:  # noqa: BLE001 - reporting must not depend on config
        provider = model = "unknown"

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provider": provider,
        "model": model,
        "ks": list(ks),
        "notes": notes,
        **summary,
        "failures": [
            {
                "suite": r.suite,
                "id": r.row_id,
                "query": r.query,
                "route": r.route,
                "degraded": r.degraded,
                "service_failure": r.service_failure,
                "reasons": r.failures,
            }
            for r in results
            if not r.passed
        ],
    }


# ---------------------------------------------------------------------------
# Deliverable D4
# ---------------------------------------------------------------------------
#: The queries transcribed into docs/sample_qa.md, by eval-set id so every entry
#: is traceable to a run. Chosen to evidence all four required behaviours: a
#: cited fact, an advice refusal, a performance refusal, and a PII rejection.
SAMPLE_QA_IDS: tuple[str, ...] = ("q01", "q06", "a01", "a06", "a09", "a15", "a02")


def write_sample_qa(
    results: Sequence[RowResult],
    out_path: Path,
    *,
    report: dict[str, Any] | None = None,
) -> int:
    """D4: the assistant's real answers, transcribed from this run.

    Nothing here is paraphrased. If a row was degraded the entry says so, because
    a deliverable that quietly shows a fallback message as if it were a real
    answer would misrepresent the system.
    """
    by_id = {r.row_id: r for r in results}
    chosen = [by_id[i] for i in SAMPLE_QA_IDS if i in by_id]

    behaviour = {
        Route.FACTUAL.value: "Cited factual answer",
        Route.ADVICE.value: "Advice refusal",
        Route.PERFORMANCE.value: "Performance refusal",
        Route.OUT_OF_SCOPE.value: "Out-of-scope refusal",
        Route.PII.value: "PII rejection",
    }

    lines: list[str] = [
        "# D4 - sample Q&A",
        "",
        "Generated by `python -m eval.run_eval --sample-qa`. Every answer below is",
        "the assistant's real output, transcribed verbatim from that run - nothing",
        "is paraphrased or hand-written.",
        "",
    ]
    if report:
        lines += [
            f"- Run at: `{report.get('generated_at', 'unknown')}`",
            f"- Provider: `{report.get('provider', 'unknown')}` / "
            f"model `{report.get('model', 'unknown')}`",
            f"- Overall pass rate: **{report.get('overall_pass_rate', 0) * 100:.1f}%** "
            f"({report.get('overall_passed', 0)}/{report.get('overall_n', 0)})",
            "",
        ]
        for note in report.get("notes", []):
            lines += [f"> **Note from the run:** {note}", ""]

    for index, result in enumerate(chosen, 1):
        if result.service_failure:
            status = "degraded - generation failed"
        elif result.degraded:
            status = "refused by the contract - the system chose not to answer"
        else:
            status = "as expected"
        lines += [
            f"## {index}. {behaviour.get(result.route, result.route)}",
            "",
            f"**Query** ({result.row_id}, route `{result.route}`):",
            "",
            f"> {result.query}",
            "",
            "**Answer:**",
            "",
            "```",
            result.answer.rstrip() or "(empty)",
            "```",
            "",
            f"*Contract checks: {'all passed' if result.passed else 'failed'} - "
            f"{status}.*",
            "",
        ]
        if result.failures:
            for reason in result.failures:
                lines.append(f"* failing check - {reason}")
            lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return len(chosen)


# ---------------------------------------------------------------------------
# Deliverable D2
# ---------------------------------------------------------------------------
def write_source_list(
    manifest_path: Path, out_path: Path
) -> int:
    """D2: one row per document, failed rows included (FR-1.5, A20)."""
    import csv

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents = manifest.get("documents", [])
    header = [
        "source_id",
        "scheme",
        "doc_type",
        "title",
        "publisher",
        "url",
        "retrieved_at",
        "sha256",
        "status",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for doc in documents:
            writer.writerow(
                [
                    doc.get("source_id", ""),
                    doc.get("scheme", ""),
                    doc.get("doc_type", ""),
                    doc.get("title", ""),
                    doc.get("publisher", ""),
                    doc.get("source_url", ""),
                    doc.get("retrieved_at", ""),
                    doc.get("sha256", ""),
                    "failed" if doc.get("thin") or not doc.get("content_chars") else "ok",
                ]
            )
    return len(documents)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m eval.run_eval",
        description="Run the golden and adversarial suites and write results.json.",
    )
    parser.add_argument("--golden", action="store_true", help="run the golden suite")
    parser.add_argument(
        "--adversarial", action="store_true", help="run the adversarial suite"
    )
    parser.add_argument(
        "--out", default=str(DEFAULT_OUT), help="where to write results JSON"
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=list(DEFAULT_KS),
        help=f"recall cutoffs (default: {' '.join(map(str, DEFAULT_KS))})",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="print each full answer and citation (this is what feeds D4)",
    )
    parser.add_argument(
        "--source-list",
        action="store_true",
        help="also regenerate docs/source_list.csv (D2)",
    )
    parser.add_argument(
        "--sample-qa",
        action="store_true",
        help="also regenerate docs/sample_qa.md (D4) from this run",
    )
    args = parser.parse_args(argv)

    run_golden = args.golden or not args.adversarial
    run_adv = args.adversarial or not args.golden
    ks = tuple(dict.fromkeys(args.k))

    from rag.pipeline import ask

    results: list[RowResult] = []
    if run_golden:
        print("=" * 72)
        print("GOLDEN SUITE - factual contract + recall")
        print("=" * 72)
        results += run_suite(
            "golden",
            load_rows(GOLDEN_PATH),
            ask,
            ks=ks,
            checker=check_golden_row,
        )
        print_table(results, show=args.show)
    if run_adv:
        print("\n" + "=" * 72)
        print("ADVERSARIAL SUITE - routing + PII")
        print("=" * 72)
        adv = run_suite(
            "adversarial",
            load_rows(ADVERSARIAL_PATH),
            ask,
            ks=ks,
            checker=check_adversarial_row,
        )
        print_table(adv, show=args.show)
        results += adv

    report = build_report(results, ks)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for name, stats in report["suites"].items():
        line = (
            f"  {name:<12} {stats['passed']}/{stats['n']} passed "
            f"({stats['pass_rate'] * 100:.1f}%)"
        )
        for key, value in stats.items():
            if key.startswith("recall@") and not key.endswith("_hits"):
                line += f"  {key}={value * 100:.0f}%"
        print(line)
        for label in ("retrieval_checks", "generation_checks"):
            tally = stats.get(label)
            if tally:
                print(
                    f"                 {label:<18} {tally['passed']}/{tally['total']} "
                    f"({tally['pass_rate'] * 100:.1f}%)"
                )
    print(
        f"  overall      {report['overall_passed']}/{report['overall_n']} "
        f"({report['overall_pass_rate'] * 100:.1f}%)"
    )
    print(
        f"  latency      p50={report['latency_ms']['p50']}ms  "
        f"p95={report['latency_ms']['p95']}ms"
    )
    for note in report["notes"]:
        print(f"\n  NOTE: {note}")
    print(f"\n  wrote {out_path}")

    if args.source_list:
        n = write_source_list(
            PROJECT_ROOT / "data" / "manifest.json",
            PROJECT_ROOT / "docs" / "source_list.csv",
        )
        print(f"  wrote docs/source_list.csv ({n} rows)")

    if args.sample_qa:
        n = write_sample_qa(
            results, PROJECT_ROOT / "docs" / "sample_qa.md", report=report
        )
        print(f"  wrote docs/sample_qa.md ({n} queries)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
