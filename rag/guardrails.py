"""Stage 0 and Stage 1 of Flow B: the PII screen and the router.

Both live here because the ordering between them is the security property, not an
implementation detail. The PII screen runs **first and deterministically**, before
any model call, so no amount of prompt persuasion can talk it past a PAN
(architecture.md §6.2, D3, invariant I4). The router runs second and is allowed
to be fuzzy, because a wrong ``FACTUAL`` label is still caught by the Stage 2/4
gates, whereas a missed PII hit is not recoverable (architecture.md §6.3).

Two safeguards stop the PII screen from flagging the corpus's own vocabulary.
Mutual fund pages are full of numbers, so a naive screen rejects most of them:

* **Keyword gating.** A bare 8-digit number is not an account number. ACCOUNT and
  OTP both require a nearby keyword, so "SIP of 5000" and "lock-in is 3 years"
  stay clean.
* **Safe-context whitelisting.** A digit run next to ``%``/``₹``/``Rs``/``min``/
  ``SIP``/``80C``, or followed by ``SIP``/``years``/``months``, is a *figure*, not
  an identifier, and cannot trigger ACCOUNT.
"""

from __future__ import annotations

import re
from functools import cache
from typing import Any, Final, Iterable, Sequence

from rag.prompts import DEFAULT_EDUCATION_URL, refusal_text
from rag.schemas import Route, RouteResult

#: The six classes, in the order :func:`detect_pii` reports them.
PII_CLASSES: Final[tuple[str, ...]] = (
    "PAN",
    "AADHAAR",
    "ACCOUNT",
    "OTP",
    "EMAIL",
    "PHONE",
)

# --------------------------------------------------------------------------
# §6.2 patterns
# --------------------------------------------------------------------------

#: PAN: five letters, four digits, one letter. Uppercase, as PANs are written.
_PAN_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")

#: Aadhaar: 12 digits, first digit 2-9, optional spaces or dashes as separators.
_AADHAAR_RE = re.compile(r"(?<![\d])[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}(?!\d)")

#: Account/folio/card: a 6-19 digit run, gated by a keyword and not in a safe
#: context. The range reaches 19 because a 13-19 digit debit/credit card number
#: is otherwise caught by *no* class at all: 16 digits cannot match the 6-12
#: account window, is too long for Aadhaar (12), and is far past OTP (4-6).
#: Letting a card number through would put it in an LLM prompt and in the logs,
#: which the PII invariants forbid.
_ACCOUNT_RE = re.compile(r"(?<![\d])\d{6,19}(?![\d])")

#: OTP: a 4-6 digit run, gated by a keyword and not in a safe context.
_OTP_RE = re.compile(r"(?<![\d])\d{4,6}(?![\d])")

_EMAIL_RE = re.compile(r"\b[\w.%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

#: Phone: optional +91 then a 10-digit number starting 6-9.
_PHONE_RE = re.compile(r"(?<![\d+])(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)")

#: Keywords that make a digit run an ACCOUNT/folio/card reference rather than a
#: figure. ``card`` and ``cvv`` are here because a 13-19 digit card number is
#: the case that no other class would catch.
_ACCOUNT_KEYWORDS: Final[tuple[str, ...]] = (
    "account",
    "a/c",
    "folio",
    "demat",
    "pan",
    "number no",
    "linked",
    "card",
    "credit",
    "debit",
    "visa",
    "mastercard",
)

#: Keywords that make a digit run an OTP.
_OTP_KEYWORDS: Final[tuple[str, ...]] = (
    "otp",
    "one time password",
    "one-time password",
    "verification code",
    "verify code",
)

#: Tokens before a digit run that mark it as an amount, rate or limit.
_SAFE_PREFIXES: Final[tuple[str, ...]] = (
    "rs",
    "rs.",
    "inr",
    "min",
    "sip",
    "80c",
    "80d",
    "per",
)

#: Tokens after a digit run that mark it as a duration, rate or amount.
_SAFE_SUFFIXES: Final[tuple[str, ...]] = (
    "sip",
    "year",
    "years",
    "yr",
    "yrs",
    "month",
    "months",
    "day",
    "days",
    "per",
)

#: Currency and percent symbols that directly precede a figure.
_SAFE_SYMBOLS: Final[tuple[str, ...]] = ("%", "₹", "rs", "inr")

#: How far from the run a keyword may sit and still count as "nearby".
_KEYWORD_WINDOW = 40


def _verhoef_check_digit(first_eleven: str) -> int:
    """Return the Verhoef check digit for the first eleven digits.

    Args:
        first_eleven: exactly eleven digits.

    Returns:
        The twelfth digit that makes the twelve-digit string check out.
    """
    c = 0
    for i, ch in enumerate(first_eleven):
        c = _D[c][_P[(i + 1) % 8][int(ch)]]
    for candidate in range(10):
        if _D[c][_P[12 % 8][candidate]] == 0:
            return candidate
    return 0


def _verhoef_valid(digits: str) -> bool:
    """Whether a twelve-digit string satisfies the Verhoef checksum.

    Args:
        digits: twelve digits, already stripped of separators.

    Returns:
        ``True`` when the check digit agrees.
    """
    c = 0
    for i, ch in enumerate(digits):
        c = _D[c][_P[(i + 1) % 8][int(ch)]]
    return c == 0


#: Verhoeff multiplication table.
_D: Final[tuple[tuple[int, ...], ...]] = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)

#: Verhoeff permutation table.
_P: Final[tuple[tuple[int, ...], ...]] = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)


def _near(text: str, start: int, end: int, keywords: Sequence[str]) -> bool:
    """Whether any keyword appears within the window around ``text[start:end]``."""
    window = text[max(0, start - _KEYWORD_WINDOW) : end + _KEYWORD_WINDOW].lower()
    return any(keyword in window for keyword in keywords)


def _safe_context(text: str, start: int, end: int) -> bool:
    """Whether a digit run is a published figure rather than an identifier.

    Checks the token immediately before and after the run, plus a currency or
    percent symbol directly before it.
    """
    before = text[:start].rstrip()
    after = text[end:].lstrip()

    tail = before[-1:].lower()
    if tail in _SAFE_SYMBOLS:
        return True
    for token in _SAFE_PREFIXES:
        if before.lower().endswith(token):
            return True
    for token in _SAFE_SUFFIXES:
        if after.lower().startswith(token):
            return True
    return False


def _digits_in_run(text: str, match: re.Match[str]) -> str:
    """Strip separators from a matched numeric run."""
    return re.sub(r"\D", "", match.group(0))


def detect_pii(text: str) -> list[str]:
    """Return the PII classes present in ``text`` (architecture.md §6.2).

    Args:
        text: the raw user query. It is read only; nothing is stored or logged.

    Returns:
        The matched class names, in :data:`PII_CLASSES` order. Empty when the
        text is clean.

    Example:
        >>> detect_pii("My PAN is ABCDE1234F")
        ['PAN']
        >>> detect_pii("minimum SIP is ₹500")
        []
    """
    if not text or not text.strip():
        return []

    found: set[str] = set()

    if _PAN_RE.search(text):
        found.add("PAN")

    for match in _AADHAAR_RE.finditer(text):
        digits = _digits_in_run(text, match)
        if len(digits) == 12 and _verhoef_valid(digits):
            found.add("AADHAAR")

    for match in _ACCOUNT_RE.finditer(text):
        if _safe_context(text, match.start(), match.end()):
            continue
        if _near(text, match.start(), match.end(), _ACCOUNT_KEYWORDS):
            found.add("ACCOUNT")
            break

    for match in _OTP_RE.finditer(text):
        if _safe_context(text, match.start(), match.end()):
            continue
        if _near(text, match.start(), match.end(), _OTP_KEYWORDS):
            found.add("OTP")
            break

    if _EMAIL_RE.search(text):
        found.add("EMAIL")

    if _PHONE_RE.search(text):
        found.add("PHONE")

    return [name for name in PII_CLASSES if name in found]


# --------------------------------------------------------------------------
# §6.3 router
# --------------------------------------------------------------------------

#: PERFORMANCE triggers, checked before ADVICE so a combined query gets the
#: factsheet redirect rather than a generic advice refusal.
_PERFORMANCE_TERMS: Final[tuple[str, ...]] = (
    "return",
    "returns",
    "cagr",
    "performance",
    "perform",
    "performed",
    "how much did it grow",
    "how much has it grown",
    "has grown",
    "grown",
    "grew",
    "best performing",
    "top performer",
    "ranking",
    "rank",
    "ranked",
    "nav trend",
    "x-year return",
    "1 year return",
    "3 year return",
    "5 year return",
    "year return",
    "yearly return",
    "gain",
    "gained",
    "loss",
    "yield so far",
    "growth",
)

#: ``Direct Growth`` is a *plan* name, not a performance claim, and all five
#: corpus pages are Direct Growth plans. Left in place, the bare ``growth``
#: trigger sent "What is the expense ratio of HDFC Large Cap Fund - Direct
#: Growth?" to PERFORMANCE, so the flagship factual question was refused before
#: retrieval ever ran. Strip plan-name usages before the keyword check; a
#: genuine growth query such as "What is the growth of HDFC Large Cap?" keeps
#: the term and still routes to PERFORMANCE.
_PLAN_NAME_RE: Final[re.Pattern[str]] = re.compile(
    r"\bdirect[\s\-_]+(?:plan[\s\-_]+)?growth\b|\bgrowth[\s\-_]+(?:option|plan|fund)\b",
    re.IGNORECASE,
)

#: ADVICE triggers.
_ADVICE_TERMS: Final[tuple[str, ...]] = (
    "should i",
    "which is best",
    "which one is best",
    "recommend",
    "suggest",
    "worth buying",
    "worth investing",
    "good time",
    "is it safe for me",
    "safe for me",
    "suitable for",
    "my portfolio",
    "allocate",
    "allocation",
    "target allocation",
    "asset allocation",
    "better than",
    "right for me",
    "good for me",
    "good investment",
    "can i buy",
    "should we buy",
    "help me choose",
    "which fund should",
)

#: §6.3 lists "which is best" as an ADVICE trigger. That literal misses the
#: natural phrasings ("which of these five funds is best?"), so the family is
#: matched as a pattern instead. PERFORMANCE is evaluated first, so
#: "which fund has the best returns" is still a PERFORMANCE question.
_ADVICE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\bwhich\b[^?]*\bbest\b"),
    re.compile(r"\bwhat\b[^?]*\bbest\b[^?]*\b(fund|scheme|one)\b"),
)

#: Labels the classifier is allowed to emit.
_CLASSIFIER_LABELS: Final[frozenset[str]] = frozenset(
    {"FACTUAL", "ADVICE", "PERFORMANCE", "OUT_OF_SCOPE"}
)

_CLASSIFIER_SYSTEM: Final[str] = (
    "You label questions about HDFC mutual fund schemes. Reply with exactly one "
    "word and nothing else: FACTUAL, ADVICE, PERFORMANCE or OUT_OF_SCOPE.\n"
    "FACTUAL = a published scheme fact (fees, exit load, benchmark, minimum "
    "amount, risk, fund manager, tax treatment).\n"
    "ADVICE = should I buy, is it right for me, which is best, portfolio or "
    "allocation guidance, suitability.\n"
    "PERFORMANCE = past or projected returns, CAGR, rankings, NAV movement, "
    "which fund grew most.\n"
    "OUT_OF_SCOPE = today's NAV, current price, live quote, app login, account "
    "opening, SIP calculators, or any fund outside the five HDFC schemes."
)


def _has_term(lowered: str, terms: Iterable[str]) -> bool:
    """Whether any trigger term appears in the lowercased query.

    Matching is on word boundaries. Plain substring matching caused real
    mis-routes, because a trigger can sit inside an unrelated word: ``gain``
    inside ``bargain`` and ``against``, ``rank`` inside ``frank``.
    """
    return any(_term_re(term).search(lowered) for term in terms)


@cache
def _term_re(term: str) -> re.Pattern[str]:
    """A whole-word pattern for a trigger term or multi-word phrase."""
    return re.compile(rf"(?<!\w){re.escape(term)}(?!\w)")


def _has_advice_pattern(lowered: str) -> bool:
    """Whether the query matches a comparative-superlative advice pattern."""
    return any(pattern.search(lowered) for pattern in _ADVICE_PATTERNS)


def _classify_with_llm(query: str, llm: Any) -> Route:
    """Ask the LLM classifier for one label, defaulting to ``FACTUAL``.

    A parse failure is not an error. ``FACTUAL`` is the safe default because the
    Stage 2 and Stage 4 gates still apply downstream, so a misclassification
    cannot produce an ungrounded answer (architecture.md §6.3).

    Args:
        query: the user query, which by this point has passed the PII screen.
        llm: an object satisfying :class:`~rag.llm_client.LLMClient`, or ``None``.

    Returns:
        The routed label.
    """
    if llm is None:
        return Route.FACTUAL

    from rag.llm_client import LLMError

    try:
        raw = llm.complete(_CLASSIFIER_SYSTEM, query, json_mode=False)
    except LLMError:
        return Route.FACTUAL
    except Exception:  # noqa: BLE001 - a classifier must not break the query
        return Route.FACTUAL

    token = str(raw).strip().strip(".").strip('"').upper().split()
    if token and token[0] in _CLASSIFIER_LABELS:
        return Route(token[0])
    return Route.FACTUAL


def route(
    query: str,
    *,
    llm: Any = None,
    allowlist: Iterable[str] | None = None,
) -> RouteResult:
    """Label a query before retrieval (architecture.md §6.3).

    Precedence is ``PII > PERFORMANCE > ADVICE > OUT_OF_SCOPE > FACTUAL``.
    ``PERFORMANCE`` is checked before ``ADVICE`` so that "which fund has the best
    returns and should I buy it" gets the factsheet redirect.

    Args:
        query: the raw user query.
        llm: optional classifier, used only after the deterministic rules miss.
        allowlist: the ingested URL set. Accepted for interface stability and to
            let a caller confirm the corpus; the deterministic screen and the
            keyword rules never consult it.

    Returns:
        A :class:`~rag.schemas.RouteResult`. For ``PII`` the caller must discard
        the text without logging or persisting it.
    """
    classes = detect_pii(query)
    if classes:
        return RouteResult(
            label=Route.PII, reason="pii", matched_classes=tuple(classes)
        )

    from rag.retrieve import resolve_scheme

    scheme_hint = resolve_scheme(query)
    lowered = query.lower()
    # Plan names are stripped first so "Direct Growth" is not read as a
    # performance claim. Everything else routes off the untouched query.
    lowered = _PLAN_NAME_RE.sub(" ", lowered)

    if _has_term(lowered, _PERFORMANCE_TERMS):
        return RouteResult(
            label=Route.PERFORMANCE,
            reason="keyword:performance",
            scheme_hint=scheme_hint,
        )

    if _has_term(lowered, _ADVICE_TERMS) or _has_advice_pattern(lowered):
        return RouteResult(
            label=Route.ADVICE, reason="keyword:advice", scheme_hint=scheme_hint
        )

    label = _classify_with_llm(query, llm)
    return RouteResult(
        label=label,
        reason="llm" if llm is not None else "default",
        scheme_hint=scheme_hint,
    )


#: The §7 PII rejection shape. Deliberately fixed text: it never echoes the input
#: and never names the class that actually matched, beyond naming the categories
#: in general. The matched classes go to the log, not to the user.
_PII_RESPONSE: Final[str] = (
    "This assistant does not accept personal or account information such as PAN, "
    "Aadhaar, account or folio numbers, OTPs, email addresses or phone numbers. "
    "Please remove it and ask a factual question about the schemes.\n"
    "Facts-only. No investment advice."
)


def pii_response(matched_classes: Sequence[str] | None = None) -> str:
    """Return the neutral PII rejection text (PRD §7).

    Args:
        matched_classes: the classes that triggered the screen. Accepted so
            callers can log them alongside the response; deliberately **not**
            rendered into the text, because telling a user which pattern matched
            is a small oracle for building one that evades it.

    Returns:
        The two-line rejection, ending in "Facts-only. No investment advice."
    """
    return _PII_RESPONSE


def refusal_payload(
    label: str,
    *,
    answer_cfg: Any = None,
    route: Route = Route.FACTUAL,
    trace: Any = None,
) -> Any:
    """Build a §7 refusal payload for a pre-retrieval route.

    Args:
        label: a key of :data:`~rag.prompts.REFUSALS`, e.g. ``"ADVICE"``.
        answer_cfg: answer settings, for the education URL and date format.
        route: the route that produced this refusal, so observability and the
            UI report ``ADVICE`` rather than a generic label.
        trace: an optional retrieval trace to attach.

    Returns:
        An :class:`~rag.schemas.AnswerPayload` in the required refusal shape.
    """
    from datetime import datetime, timezone

    from rag.schemas import AnswerPayload
    from rag.validate import _format_retrieved_at

    education_url = getattr(answer_cfg, "education_url", DEFAULT_EDUCATION_URL)
    date_format = getattr(answer_cfg, "date_format", "%d %b %Y")
    freshness_label = getattr(answer_cfg, "freshness_label", "Last updated from sources:")

    text, url = refusal_text(label, education_url=education_url)
    stamp = _format_retrieved_at(datetime.now(timezone.utc).isoformat(), date_format)

    lines = [text, ""]
    if url:
        lines.append(f"Source: {url}")
    lines.append(f"{freshness_label} {stamp}".rstrip())

    return AnswerPayload(
        text="\n".join(lines),
        route=route,
        source_url=url,
        grounded_field=None,
        trace=trace,
        degraded=False,
        validators_passed=[],
    )


__all__ = [
    "PII_CLASSES",
    "detect_pii",
    "pii_response",
    "refusal_payload",
    "route",
]
