"""Pluggable extractors behind one interface.

``RuleBasedExtractor`` - the frozen v1 baseline (regex gazetteer + ConText).
``LLMExtractor``       - an open-weight instruction-tuned model (Qwen / Llama /
                         Mistral, <= 8B) that fills a schema, followed by the
                         deterministic :class:`FindingVerifier`.

Both return :class:`ExtractionResult`, so the pipeline, database, triage,
evaluator and dashboard are identical for either system.
"""

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from .context import ConTextEngine
from .extractor import ClinicalExtractor
from .llm.backends import BackendError, LLMBackend, ResponseCache
from .llm.prompts import PROMPT_VERSION, PromptBuilder
from .schemas import Finding
from .sectionizer import SectionedReport
from .verification import FindingVerifier

LOGGER = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    findings: list[Finding]
    schema_errors: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


class BaseExtractor(ABC):
    name: str = "base"

    @abstractmethod
    def extract(self, report_id: str, text: str, sectioned: SectionedReport) -> ExtractionResult:
        """Extract findings from de-identified ``text`` (``sectioned`` is its sectionizer view)."""

    def describe(self) -> dict[str, str]:
        return {"extractor": self.name}


class RuleBasedExtractor(BaseExtractor):
    name = "rules"

    def __init__(self) -> None:
        self.engine = ClinicalExtractor(ConTextEngine())

    def extract(self, report_id: str, text: str, sectioned: SectionedReport) -> ExtractionResult:
        findings, errors = self.engine.extract(report_id, sectioned)
        return ExtractionResult(findings, errors, {"proposed": len(findings) + len(errors)})


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I | re.M)


def parse_model_json(raw: str) -> list[dict]:
    """Tolerant JSON parsing: strips code fences / chatter, accepts {"findings": [...]} or a bare list."""
    cleaned = _FENCE.sub("", raw.strip())
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start = min([i for i in (cleaned.find("{"), cleaned.find("[")) if i >= 0], default=-1)
        end = max(cleaned.rfind("}"), cleaned.rfind("]"))
        if start < 0 or end <= start:
            raise ValueError("no JSON object in model output")
        data = json.loads(cleaned[start:end + 1])
    if isinstance(data, dict):
        data = data.get("findings", data.get("Findings"))
    if not isinstance(data, list):
        raise ValueError('expected {"findings": [...]}')
    return data


class LLMExtractor(BaseExtractor):
    name = "llm"

    def __init__(self, backend: LLMBackend, prompts: PromptBuilder, verifier: Optional[FindingVerifier] = None,
                 cache: Optional[ResponseCache] = None, max_retries: int = 2, use_schema: bool = True) -> None:
        self.backend = backend
        self.prompts = prompts
        self.verifier = verifier or FindingVerifier()
        self.cache = cache
        self.max_retries = max_retries
        self.use_schema = use_schema

    @property
    def run_name(self) -> str:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", self.backend.identity.split(":", 1)[-1]).strip("-")
        return f"llm_{slug}"

    def describe(self) -> dict[str, str]:
        return {"extractor": self.name, "backend": self.backend.name, "model": self.backend.identity,
                "prompt_version": PROMPT_VERSION, "few_shot_ids": ",".join(e.report_id for e in self.prompts.examples)}

    def _call(self, messages: list[dict[str, str]], schema: Optional[dict]) -> tuple[str, bool]:
        key = ResponseCache.key(self.backend.identity, messages, schema)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return hit, True
        raw = self.backend.complete(messages, schema)
        if self.cache is not None:
            self.cache.put(key, raw, {"model": self.backend.identity, "prompt_version": PROMPT_VERSION})
        return raw, False

    def extract(self, report_id: str, text: str, sectioned: SectionedReport) -> ExtractionResult:
        schema = self.prompts.output_schema() if self.use_schema else None
        messages = self.prompts.build(text, exclude_report_id=report_id)
        diag: dict[str, Any] = {"attempts": 0, "cache_hits": 0, "latency_s": 0.0, "parse_errors": []}
        items: Optional[list[dict]] = None
        t0 = time.perf_counter()
        for attempt in range(self.max_retries + 1):
            diag["attempts"] = attempt + 1
            try:
                raw, cached = self._call(messages, schema)
            except BackendError as exc:
                diag["backend_error"] = str(exc)
                LOGGER.error("%s: backend error: %s", report_id, exc)
                break
            diag["cache_hits"] += int(cached)
            try:
                items = parse_model_json(raw)
                break
            except (ValueError, json.JSONDecodeError) as exc:
                diag["parse_errors"].append(str(exc))
                messages = messages + [
                    {"role": "assistant", "content": raw[:4000]},
                    {"role": "user", "content": f"That output was not valid ({exc}). Reply with only the JSON "
                                                'object {"findings": [...]} following the schema.'},
                ]
        diag["latency_s"] = time.perf_counter() - t0
        if items is None:
            return ExtractionResult([], [f"{report_id}: no parsable model output"], diag | {"proposed": 0})
        verified = self.verifier.verify(report_id, text, items, extractor=self.run_name)
        diag.update(proposed=verified.proposed, rejected=verified.rejected)
        return ExtractionResult(verified.findings, verified.schema_errors, diag)
