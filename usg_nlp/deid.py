"""Stage 1a - HIPAA Safe Harbor de-identification audit.

The scanner targets the Safe Harbor direct-identifier classes that realistically
occur in Indian and US radiology narratives: names (titled or labelled),
calendar dates, phone/fax numbers, e-mail addresses, medical record /
hospital / accession numbers, national ID numbers (Aadhaar / SSN pattern),
URLs and IP addresses, and ages above 89.

Design notes
------------
* Patterns are anchored on context (a label such as ``MRN:`` or a title such as
  ``Dr.``) wherever a bare pattern would collide with clinical text.  A bare
  ``\\d+\\.\\d+`` would otherwise flag every "8.0 mm" measurement.
* Overlapping hits are merged, the longest span winning, and replaced by a typed
  placeholder (``[DATE]``, ``[MRN]``...) so downstream NLP keeps sentence shape.
* The audit never mutates the clinical content outside identifier spans.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .schemas import DeidentificationResult, PHIEntity

_MONTHS = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
    r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
# Title-case tokens joined by single spaces; an all-caps label such as "MRN" ends the name.
_NAME = r"[A-Z][a-z'\-]+(?:[ \t](?:[A-Z]\.|[A-Z][a-z'\-]+)){0,3}"


@dataclass(frozen=True)
class _PHIPattern:
    phi_type: str
    regex: re.Pattern[str]
    group: int = 0  # capture group holding the identifier (0 = whole match)


_PATTERNS: tuple[_PHIPattern, ...] = (
    _PHIPattern("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    _PHIPattern("URL", re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.I)),
    _PHIPattern("IP_ADDRESS", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    _PHIPattern(
        "MRN",
        re.compile(
            r"\b(?:MRN|UHID|CR\s*No|IP\s*No|OP\s*No|Patient\s*ID|Pt\.?\s*ID|Reg(?:istration)?\.?\s*No|"
            r"Hospital\s*No|Accession(?:\s*No)?|Lab\s*No|Bill\s*No|Ref(?:erence)?\s*No)\.?\s*[:#\-]?\s*"
            r"([A-Z0-9][A-Z0-9/\-]{3,})",
            re.I,
        ),
        group=1,
    ),
    _PHIPattern("NATIONAL_ID", re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b")),  # Aadhaar layout
    _PHIPattern("NATIONAL_ID", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),  # SSN layout
    _PHIPattern("PHONE", re.compile(r"(?<!\d)(?:\+91[\s\-]?|0)?[6-9]\d{4}[\s\-]?\d{5}(?!\d)")),
    _PHIPattern("PHONE", re.compile(r"(?<!\d)(?:\+1[\s\-]?)?\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}(?!\d)")),
    _PHIPattern("DATE", re.compile(r"\b\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}\b")),
    _PHIPattern("DATE", re.compile(r"\b\d{1,2}[/\-.]\d{1,2}[/\-.](?:\d{4}|\d{2})\b")),
    _PHIPattern("DATE", re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?[\s\-]+{_MONTHS}[\s\-,]+\d{{2,4}}\b", re.I)),
    _PHIPattern("DATE", re.compile(rf"\b{_MONTHS}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b", re.I)),
    _PHIPattern(
        "NAME",
        re.compile(
            rf"\b(?:Patient(?:'s)?\s*Name|Pt\.?\s*Name|Name|Patient|Referred\s+by|Ref\.?\s*by|"
            rf"Reported\s+by|Sonologist|Radiologist|Consultant)\s*[:\-]\s*((?:(?:Dr|Mr|Mrs|Ms|Miss)\.?\s+)?{_NAME})"
        ),
        group=1,
    ),
    _PHIPattern("NAME", re.compile(rf"\b(?:Dr|Mr|Mrs|Ms|Miss|Prof)\.?\s+{_NAME}")),
    _PHIPattern("AGE_OVER_89", re.compile(r"\b(?:9\d|1[0-1]\d)\s*(?:-\s*)?(?:years?|yrs?|y/o|yo)\b", re.I)),
)


class PHIScanner:
    """Detects and redacts direct identifiers in free-text reports."""

    def __init__(self, patterns: tuple[_PHIPattern, ...] = _PATTERNS) -> None:
        self._patterns = patterns

    def detect(self, text: str) -> list[PHIEntity]:
        raw: list[tuple[int, int, str]] = []
        for pattern in self._patterns:
            for match in pattern.regex.finditer(text):
                start, end = match.span(pattern.group)
                if end > start:
                    raw.append((start, end, pattern.phi_type))
        return [PHIEntity(phi_type=t, text=text[s:e], start=s, end=e) for s, e, t in self._merge(raw)]

    @staticmethod
    def _merge(spans: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
        """Merge overlapping spans; on overlap the longest span keeps its type."""
        merged: list[tuple[int, int, str]] = []
        for start, end, phi_type in sorted(spans, key=lambda s: (s[0], -(s[1] - s[0]))):
            if merged and start < merged[-1][1]:
                prev_start, prev_end, prev_type = merged[-1]
                keep_type = prev_type if (prev_end - prev_start) >= (end - start) else phi_type
                merged[-1] = (prev_start, max(prev_end, end), keep_type)
            else:
                merged.append((start, end, phi_type))
        return merged

    def redact(self, report_id: str, text: str) -> DeidentificationResult:
        entities = self.detect(text)
        pieces: list[str] = []
        cursor = 0
        for ent in entities:
            pieces.append(text[cursor : ent.start])
            pieces.append(f"[{ent.phi_type}]")
            cursor = ent.end
        pieces.append(text[cursor:])
        return DeidentificationResult(report_id=report_id, phi_entities=entities, redacted_text="".join(pieces))
