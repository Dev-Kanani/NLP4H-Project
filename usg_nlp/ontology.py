"""Ontology loader: the single source of truth for vocabulary, codes and triage policy.

Both extractors (the rule baseline and the LLM extractor), the verification
layer, the triage classifier and the evaluator read the same
``resources/ontology.json``.  Adding a finding is a data change, not a code
change.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

ONTOLOGY_PATH = Path(__file__).resolve().parent / "resources" / "ontology.json"
_LATERALITY_SUFFIX = re.compile(r"\s*-\s*(left|right|bilateral)$", re.I)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


@dataclass(frozen=True)
class ConceptEntry:
    organs: tuple[str, ...]
    finding: str
    concept: str
    pathological: bool
    sizeable: bool
    definition: str
    radlex: Optional[str]
    icd10: Optional[str]
    tier: object  # int, or {grade: int} for steatosis
    reason: str
    aliases: tuple[str, ...] = ()
    legacy: bool = False

    def tier_for(self, grade: str) -> int:
        if isinstance(self.tier, dict):
            return int(self.tier.get(grade, self.tier.get("N/A", 2)))
        return int(self.tier)


@dataclass
class Ontology:
    version: str
    organ_aliases: dict[str, tuple[str, ...]]
    entries: list[ConceptEntry]
    _index: dict[tuple[str, str], ConceptEntry] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for e in self.entries:
            for organ in e.organs:
                self._index.setdefault((organ, e.finding), e)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_file(cls, path: Path = ONTOLOGY_PATH) -> "Ontology":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        entries = [
            ConceptEntry(
                organs=tuple(c["organs"]), finding=c["finding"], concept=c["concept"],
                pathological=bool(c["pathological"]), sizeable=bool(c["sizeable"]),
                definition=c.get("definition", ""), radlex=c.get("radlex"), icd10=c.get("icd10"),
                tier=c.get("tier", 1), reason=c.get("reason", c["finding"].lower()),
                aliases=tuple(c.get("aliases", [])), legacy=bool(c.get("legacy", False)),
            )
            for c in raw["concepts"]
        ]
        organs = {name: tuple(spec.get("aliases", [])) for name, spec in raw["organs"].items()}
        return cls(version=raw["version"], organ_aliases=organs, entries=entries)

    # ------------------------------------------------------------------ lookups
    @property
    def organs(self) -> list[str]:
        return list(self.organ_aliases)

    def get(self, organ: str, finding: str) -> Optional[ConceptEntry]:
        return self._index.get((organ, finding))

    def vocabulary(self, organ: str, include_legacy: bool = False) -> list[str]:
        return [e.finding for e in self.entries if organ in e.organs and (include_legacy or not e.legacy)]

    def all_findings(self, include_legacy: bool = False) -> list[str]:
        seen: dict[str, None] = {}
        for e in self.entries:
            if include_legacy or not e.legacy:
                seen.setdefault(e.finding, None)
        return list(seen)

    def concept_of(self, organ: str, finding: str) -> str:
        """Normalised concept for a (possibly v1-style) finding string, e.g. 'Cortical Cyst - Right'."""
        base = _LATERALITY_SUFFIX.sub("", str(finding)).strip()
        entry = self.get(organ, base) or self.get("Global", base)
        if entry is None:  # organ mismatch (e.g. v1 files 'Unremarkable Study' under Pelvis)
            entry = next((e for e in self.entries if e.finding == base), None)
        return entry.concept if entry else base

    def study_level(self, finding: str) -> bool:
        entry = self.get("Global", finding)
        return entry is not None

    # ------------------------------------------------------------------ normalisation (LLM output)
    def normalize_organ(self, raw: str) -> Optional[str]:
        key = _norm(raw)
        for organ, aliases in self.organ_aliases.items():
            if key == _norm(organ) or key in {_norm(a) for a in aliases}:
                return organ
        candidates = {(_norm(a)): organ for organ, al in self.organ_aliases.items() for a in (organ, *al)}
        match = difflib.get_close_matches(key, list(candidates), n=1, cutoff=0.8)
        return candidates[match[0]] if match else None

    def normalize_finding(self, organ: str, raw: str) -> tuple[Optional[str], Optional[str], str]:
        """Map a model-produced (organ, finding) pair onto the vocabulary.

        Returns ``(organ, finding, method)``; ``finding`` is ``None`` when nothing
        matches.  Resolution order: exact name in organ -> alias in organ -> fuzzy
        name/alias in organ -> unique exact/alias match in another organ.
        """
        key = _norm(_LATERALITY_SUFFIX.sub("", raw))
        legacy = next((e for e in self.entries if e.legacy and organ in e.organs and key == _norm(e.finding)), None)
        if legacy is not None:  # v1 surface variant -> current name with the same concept
            current = next((e for e in self.entries if not e.legacy and organ in e.organs
                            and e.concept == legacy.concept), None)
            if current is not None:
                return organ, current.finding, "legacy"
        in_organ = [e for e in self.entries if organ in e.organs and not e.legacy]
        for e in in_organ:
            if key == _norm(e.finding):
                return organ, e.finding, "exact"
        for e in in_organ:
            if key in {_norm(a) for a in e.aliases}:
                return organ, e.finding, "alias"
        surface = {_norm(s): e.finding for e in in_organ for s in (e.finding, *e.aliases)}
        match = difflib.get_close_matches(key, list(surface), n=1, cutoff=0.85)
        if match:
            return organ, surface[match[0]], "fuzzy"
        elsewhere = [(o, e.finding) for e in self.entries if not e.legacy for o in e.organs
                     if o != organ and key in {_norm(e.finding), *(_norm(a) for a in e.aliases)}]
        if len({x[0] for x in elsewhere}) == 1:
            return elsewhere[0][0], elsewhere[0][1], "organ-corrected"
        return organ, None, "unmatched"

    # ------------------------------------------------------------------ grounding & policy
    def codes(self, organ: str, finding: str, status: str) -> tuple[Optional[str], Optional[str]]:
        entry = self.get(organ, finding)
        if entry is None:
            return None, None
        return entry.radlex, (entry.icd10 if status == "Affirmed" else None)

    def triage(self, organ: str, finding: str, status: str, grade: str) -> Optional[tuple[int, str]]:
        """(tier, reason) for findings above tier 1, else ``None``."""
        entry = self.get(organ, finding) or next((e for e in self.entries if e.finding == finding), None)
        if entry is None or status == "Negated":
            return None
        if status == "Possible":
            return (3, f"indeterminate {entry.reason} (cannot be excluded)") if entry.pathological else None
        tier = entry.tier_for(grade)
        if tier <= 1:
            return None
        label = f"{grade} {entry.reason}" if entry.concept == "Steatosis" and grade != "N/A" else entry.reason
        return tier, label[0].upper() + label[1:]

    def is_pathological(self, organ: str, finding: str) -> bool:
        entry = self.get(organ, finding) or next((e for e in self.entries if e.finding == finding), None)
        return bool(entry and entry.pathological)

    def is_sizeable(self, organ: str, finding: str) -> bool:
        entry = self.get(organ, finding)
        return bool(entry and entry.sizeable)


@lru_cache(maxsize=1)
def load_ontology() -> Ontology:
    return Ontology.from_file()
