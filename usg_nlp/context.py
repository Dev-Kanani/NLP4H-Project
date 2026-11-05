"""ConText / NegEx assertion engine.

A dependency-free implementation of the ConText algorithm (Harkema et al.,
2009) with the semantics used by ``medspaCy``'s ``ConTextComponent``:

* **Modifier triggers** carry a category (NEGATED / POSSIBLE) and a direction:
  ``FORWARD`` (pre-trigger, "no ___"), ``BACKWARD`` (post-trigger,
  "___ is not seen"), or ``BIDIRECTIONAL``.
* **Pseudo-triggers** ("no significant change") occupy their span so that the
  embedded cue never fires.
* **Termination terms** ("but", "however", "despite", ";") close every scope.
* A **maximum scope** (in word tokens) bounds each trigger, which keeps
  coordinated lists ("no calculus, wall thickening, or pericholecystic fluid")
  inside scope while preventing long-range leakage.

When several triggers reach a target the *nearest* one wins.  This resolves
the canonical hedging pattern *"although no definite shadowing calculus is
demonstrated, a tiny renal calculus cannot be entirely excluded"*: the second
"calculus" is bound to the adjacent post-trigger (POSSIBLE), not to the distant
"no".  Implemented natively so the baseline stays CPU-only and runs in
milliseconds; the trigger table can be exported to medspaCy rule format via
:meth:`ConTextEngine.to_medspacy_rules` for side-by-side comparison.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from .schemas import Assertion


class Direction(str, Enum):
    FORWARD = "FORWARD"
    BACKWARD = "BACKWARD"
    BIDIRECTIONAL = "BIDIRECTIONAL"
    TERMINATE = "TERMINATE"
    PSEUDO = "PSEUDO"


@dataclass(frozen=True)
class TriggerRule:
    literal: str  # regex source
    category: str  # NEGATED | POSSIBLE | TERMINATE | PSEUDO
    direction: Direction


@dataclass(frozen=True)
class TriggerHit:
    start: int
    end: int
    rule: TriggerRule


_FWD, _BWD, _BI = Direction.FORWARD, Direction.BACKWARD, Direction.BIDIRECTIONAL

DEFAULT_RULES: tuple[TriggerRule, ...] = (
    # --- pseudo-triggers (must be matched before their embedded cues)
    *(TriggerRule(p, "PSEUDO", Direction.PSEUDO) for p in (
        r"no\s+(?:significant\s+)?(?:interval\s+)?change", r"not\s+only", r"no\s+increase",
        r"gram[\s\-]negative", r"not\s+necessarily", r"without\s+difficulty",
    )),
    # --- uncertainty, post-triggers
    *(TriggerRule(p, "POSSIBLE", _BWD) for p in (
        r"cannot\s+be\s+(?:entirely\s+|completely\s+|totally\s+|definitely\s+)?(?:excluded|ruled\s+out)",
        r"can\s*not\s+be\s+(?:entirely\s+|completely\s+)?(?:excluded|ruled\s+out)",
        r"(?:is|are)\s+not\s+(?:entirely\s+)?excluded", r"(?:is|are)\s+(?:suspected|questionable|equivocal|possible|likely)",
        r"may\s+be\s+present", r"\?",
    )),
    # --- uncertainty, pre-triggers
    *(TriggerRule(p, "POSSIBLE", _FWD) for p in (
        r"possible", r"possibly", r"probable", r"probably", r"questionable", r"suspicious\s+(?:for|of)",
        r"suspected", r"may\s+represent", r"could\s+represent", r"cannot\s+exclude", r"can'?t\s+exclude",
        r"equivocal", r"indeterminate", r"r/o", r"rule\s+out",
    )),
    # --- negation, post-triggers
    *(TriggerRule(p, "NEGATED", _BWD) for p in (
        r"(?:is|are|was|were)\s+not\s+(?:seen|identified|demonstrated|visualized|visualised|detected|noted|present|evident)",
        r"(?:is|are|was|were)\s+not\s+(?:dilated|thickened|enlarged)",
        r"(?:is|are)\s+normal\s+in\s+(?:thickness|calibre|caliber)",
        r"(?:is|are)\s+(?:negative|absent)", r"\-ve\b", r"ruled\s+out", r"(?:is|are)\s+excluded",
    )),
    # --- negation, pre-triggers
    *(TriggerRule(p, "NEGATED", _FWD) for p in (
        r"no\s+(?:sonographic\s+|definite\s+|obvious\s+)?evidence\s+of", r"absence\s+of", r"negative\s+for",
        r"free\s+of", r"without", r"\bno\b", r"\bnot\b", r"\bnor\b", r"\bnil\b", r"\bnegative\b",
    )),
    # --- termination terms
    *(TriggerRule(p, "TERMINATE", Direction.TERMINATE) for p in (
        r"\bbut\b", r"\bhowever\b", r"\balthough\b", r"\bthough\b", r"\bdespite\b", r"\bwhereas\b",
        r"\bexcept\b", r"\baside\s+from\b", r"\bapart\s+from\b", r"\bwhich\b", r"\byet\b", r";",
        r"\bconsistent\s+with\b", r"\bcompatible\s+with\b", r"\bin\s+keeping\s+with\b",
    )),
)

_WORD = re.compile(r"[A-Za-z0-9]+")


class ConTextEngine:
    def __init__(self, rules: tuple[TriggerRule, ...] = DEFAULT_RULES, max_scope_words: int = 8) -> None:
        self.rules = rules
        self.max_scope_words = max_scope_words
        self._compiled = [(re.compile(r.literal, re.I), r) for r in rules]

    # ------------------------------------------------------------------ trigger detection
    def find_triggers(self, text: str) -> list[TriggerHit]:
        candidates: list[TriggerHit] = []
        for regex, rule in self._compiled:
            for m in regex.finditer(text):
                if m.end() > m.start():
                    candidates.append(TriggerHit(m.start(), m.end(), rule))
        # longest-match-first, non-overlapping (pseudo-triggers therefore shadow their cues)
        candidates.sort(key=lambda h: (-(h.end - h.start), h.start))
        accepted: list[TriggerHit] = []
        for hit in candidates:
            if all(hit.end <= a.start or hit.start >= a.end for a in accepted):
                accepted.append(hit)
        return sorted(accepted, key=lambda h: h.start)

    # ------------------------------------------------------------------ assertion
    def _words_between(self, text: str, start: int, end: int) -> int:
        return len(_WORD.findall(text[start:end])) if end > start else 0

    def assert_target(
        self, text: str, target_start: int, target_end: int, triggers: list[TriggerHit] | None = None
    ) -> tuple[Assertion, TriggerHit | None]:
        """Assertion status for the target span ``text[target_start:target_end]``."""
        hits = self.find_triggers(text) if triggers is None else triggers
        terminators = [h for h in hits if h.rule.direction is Direction.TERMINATE]
        best: tuple[int, TriggerHit] | None = None

        for hit in hits:
            if hit.rule.category not in {"NEGATED", "POSSIBLE"}:
                continue
            if hit.start >= target_start and hit.end <= target_end:
                continue  # trigger embedded in target, e.g. "definite" - never self-modifying
            forward_ok = hit.rule.direction in (Direction.FORWARD, Direction.BIDIRECTIONAL) and hit.end <= target_start
            backward_ok = hit.rule.direction in (Direction.BACKWARD, Direction.BIDIRECTIONAL) and hit.start >= target_end
            if forward_ok:
                gap_start, gap_end = hit.end, target_start
            elif backward_ok:
                gap_start, gap_end = target_end, hit.start
            else:
                continue
            if any(t.start >= gap_start and t.end <= gap_end for t in terminators):
                continue
            if self._words_between(text, gap_start, gap_end) > self.max_scope_words:
                continue
            distance = gap_end - gap_start
            if best is None or distance < best[0]:
                best = (distance, hit)

        if best is None:
            return Assertion.AFFIRMED, None
        hit = best[1]
        return (Assertion.NEGATED if hit.rule.category == "NEGATED" else Assertion.POSSIBLE), hit

    # ------------------------------------------------------------------ interoperability
    def to_medspacy_rules(self) -> list[dict[str, str]]:
        """Export the trigger table in medspaCy ``ConTextRule`` JSON shape."""
        category_map = {"NEGATED": "NEGATED_EXISTENCE", "POSSIBLE": "POSSIBLE_EXISTENCE",
                        "TERMINATE": "TERMINATE", "PSEUDO": "PSEUDO"}
        return [
            {"literal": r.literal, "category": category_map[r.category], "direction": r.direction.value}
            for r in self.rules
        ]
