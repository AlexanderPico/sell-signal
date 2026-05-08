from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from sell_signal.config import Settings
from sell_signal.prioritize import assign_priority
from sell_signal.schema import AnalysisResult, IdentifiedItem, PriceBand, PrioritizedItem

IDENTIFY_TEXT_PROMPT = """Return ONLY valid JSON array.
Identify the resale item(s) described in this text.
For each item return an object with keys:
- name
- category
- confidence
- condition_guess
- identifiers
- notes

Text:
{text}
"""

IDENTIFY_IMAGE_PROMPT = """Return ONLY valid JSON array.
Identify visible resale item(s) in this image.
For each item return an object with keys:
- name
- category
- confidence
- condition_guess
- identifiers
- notes

If there is one obvious item, return one array element.
Use lowercase category labels where possible.
"""

PRICE_RESEARCH_PROMPT = """Use web research to estimate likely resale market ranges in USD.
Focus on practical seller triage, not collector-grade precision.
Return ONLY valid JSON object with keys:
- used_low
- used_high
- used_median
- new_low
- new_high
- new_median
- currency
- evidence

The evidence field must be a JSON array of 2 to 5 short strings.
Item JSON:
{item_json}
"""


class SmartProvider:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def analyze_text(self, text: str) -> AnalysisResult:
        raw_items = self._run_json_query(IDENTIFY_TEXT_PROMPT.format(text=text))
        items = [
            self._build_prioritized_item(entry)
            for entry in self._normalize_items(raw_items)
        ]
        return AnalysisResult(
            items=items,
            provider=self.settings.provider_mode,
            model=self.settings.model,
        )

    def analyze_images(self, image_paths: list[Path]) -> AnalysisResult:
        prioritized: list[PrioritizedItem] = []
        warnings: list[str] = []
        for image_path in image_paths:
            try:
                raw_items = self._run_json_query(
                    IDENTIFY_IMAGE_PROMPT,
                    image_path=image_path,
                )
            except Exception as exc:
                warnings.append(f"{image_path.name}: {exc}")
                continue
            for entry in self._normalize_items(raw_items):
                prioritized.append(
                    self._build_prioritized_item(entry, source_image=image_path.name)
                )
        return AnalysisResult(
            items=self._merge_duplicate_items(prioritized),
            provider=self.settings.provider_mode,
            model=self.settings.model,
            warnings=warnings,
        )

    def _build_prioritized_item(
        self,
        payload: dict[str, Any],
        *,
        source_image: str | None = None,
    ) -> PrioritizedItem:
        item = IdentifiedItem.model_validate(self._normalize_identified_item(payload))
        pricing = self._research_price(item)
        prioritized = assign_priority(PrioritizedItem(item=item, pricing=pricing))
        if source_image:
            prioritized.source_images = [source_image]
        return prioritized

    def _research_price(self, item: IdentifiedItem) -> PriceBand:
        payload = self._run_json_query(
            PRICE_RESEARCH_PROMPT.format(
                item_json=json.dumps(item.model_dump(mode="json"))
            ),
            toolsets="web",
        )
        normalized = dict(payload)
        normalized.setdefault("currency", "USD")
        normalized["evidence"] = [
            str(entry) for entry in normalized.get("evidence", [])
        ]
        return PriceBand.model_validate(normalized)

    def _normalize_items(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, dict):
            return [payload]
        if isinstance(payload, list):
            return [entry for entry in payload if isinstance(entry, dict)]
        raise ValueError(
            "Expected JSON object or array from Hermes, "
            f"got: {type(payload).__name__}"
        )

    def _normalize_identified_item(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(payload)
        normalized["category"] = (
            str(normalized.get("category", "other")).strip().lower() or "other"
        )
        normalized["confidence"] = float(normalized.get("confidence", 0.0) or 0.0)
        identifiers = normalized.get("identifiers") or {}
        if not isinstance(identifiers, dict):
            identifiers = {"raw": str(identifiers)}
        normalized["identifiers"] = {
            str(key): self._stringify_value(value)
            for key, value in identifiers.items()
        }
        return normalized

    def _merge_duplicate_items(
        self,
        items: list[PrioritizedItem],
    ) -> list[PrioritizedItem]:
        merged: dict[tuple[str, str], PrioritizedItem] = {}
        for item in items:
            key = self._item_dedupe_key(item)
            existing = merged.get(key)
            if existing is None:
                merged[key] = item
                continue
            existing.source_images = sorted(
                set(existing.source_images).union(item.source_images)
            )
            existing.item.confidence = max(existing.item.confidence, item.item.confidence)
            if len(item.why) > len(existing.why):
                existing.why = item.why
            if item.priority_score > existing.priority_score:
                existing.priority_score = item.priority_score
                existing.priority_label = item.priority_label
            has_better_evidence = len(item.pricing.evidence) > len(
                existing.pricing.evidence
            )
            if item.pricing.evidence and has_better_evidence:
                existing.pricing = item.pricing
        return sorted(merged.values(), key=lambda row: row.priority_score, reverse=True)

    @staticmethod
    def _item_dedupe_key(item: PrioritizedItem) -> tuple[str, str]:
        return (
            item.item.category.strip().lower(),
            " ".join(item.item.name.strip().lower().split()),
        )

    def _run_json_query(
        self,
        prompt: str,
        *,
        image_path: Path | None = None,
        toolsets: str | None = None,
    ) -> Any:
        command = [
            self.settings.hermes_command,
            "chat",
            "-Q",
            "--source",
            "tool",
            "--ignore-rules",
            "-q",
            prompt,
        ]
        if self.settings.model:
            command.extend(["-m", self.settings.model])
        if self.settings.hermes_provider:
            command.extend(["--provider", self.settings.hermes_provider])
        if toolsets:
            command.extend(["-t", toolsets])
        if image_path is not None:
            command.extend(["--image", str(image_path)])

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                timeout=self.settings.hermes_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Hermes request timed out after {self.settings.hermes_timeout_seconds} seconds"
            ) from exc
        except subprocess.CalledProcessError as exc:
            details = (exc.stderr or exc.stdout or "").strip()
            if details:
                raise RuntimeError(details) from exc
            raise RuntimeError(
                f"Hermes command failed with exit code {exc.returncode}"
            ) from exc
        return self._parse_json_payload(result.stdout)

    @staticmethod
    def _parse_json_payload(output: str) -> Any:
        lines = [line for line in output.splitlines() if line.strip()]
        if lines and lines[0].startswith("session_id:"):
            lines = lines[1:]
        cleaned = "\n".join(lines).strip()
        if not cleaned:
            raise ValueError("Hermes returned empty output")

        start_positions = [
            pos for pos in (cleaned.find("{"), cleaned.find("[")) if pos != -1
        ]
        if not start_positions:
            raise ValueError(f"No JSON object found in Hermes output: {cleaned[:300]}")
        start = min(start_positions)
        end_obj = cleaned.rfind("}")
        end_arr = cleaned.rfind("]")
        end = max(end_obj, end_arr)
        if end < start:
            raise ValueError(f"Incomplete JSON in Hermes output: {cleaned[:300]}")
        return json.loads(cleaned[start : end + 1])

    @staticmethod
    def _stringify_value(value: Any) -> str:
        if isinstance(value, list):
            return ", ".join(str(entry) for entry in value)
        if value is None:
            return ""
        return str(value)
