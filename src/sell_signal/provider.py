from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Callable

from sell_signal.config import Settings
from sell_signal.prioritize import assign_priority
from sell_signal.schema import AnalysisResult, IdentifiedItem, PriceBand, PrioritizedItem

ProgressCallback = Callable[[str, str], None]
ItemCallback = Callable[[PrioritizedItem], None]

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
Identify distinct individually resellable items visible in this image.
For each item return an object with keys:
- name
- category
- confidence
- condition_guess
- identifiers
- notes

Rules:
- Return one array element per distinct sellable item.
- Do NOT collapse multiple books/items into a single lot, bundle, shelf, or assorted set.
- If the image shows many books on a shelf, list the individual titles you can read reliably.
- If exact titles are not readable, omit that item instead of inventing details.
- Use lowercase category labels where possible.
"""

ITEMIZE_IMAGE_PROMPT = """Return ONLY valid JSON array.
The first pass summarized this image too broadly as a grouped result.
Re-examine the image and list distinct individually resellable items only.
Do NOT return a single lot, bundle, shelf, mixed set, collection, or assorted grouping.
Return one array element per visible item that could plausibly become its own sale listing.
If exact titles are not readable, omit that item instead of guessing.
Use lowercase category labels where possible.
Context from the grouped pass:
{grouped_item_json}
"""

MEDIA_SHELF_PROMPT = """Return ONLY valid JSON array.
Focus only on shelves of books, dvds, blu-rays, or similar spine-out media.
Ignore furniture, decor, rugs, plants, and other non-media objects.
For each visible media item return an object with keys:
- name
- category
- confidence
- condition_guess
- identifiers
- notes

Rules:
- Return one array element per distinct media item.
- Prefer readable book titles and dvd/blu-ray titles visible on spines or covers.
- Use category "book" for books and "dvd" for dvds, blu-rays, and boxed video media.
- If a title is not readable reliably, omit it instead of guessing.
- If there are no clearly readable shelf-media items, return [].
"""

MEDIA_SHELF_SECTION_PROMPT = """Return ONLY valid JSON array.
Focus only on shelves of books, dvds, blu-rays, or similar spine-out media.
Ignore furniture, decor, rugs, plants, and other non-media objects.
Focus only on this section of the image: {section_name}.
For each visible media item return an object with keys:
- name
- category
- confidence
- condition_guess
- identifiers
- notes

Rules:
- Return one array element per distinct media item.
- Prefer readable book titles and dvd/blu-ray titles visible on spines or covers.
- Use category "book" for books and "dvd" for dvds, blu-rays, and boxed video media.
- If a title is not readable reliably, omit it instead of guessing.
- If there are no clearly readable shelf-media items in this section, return [].
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

    def analyze_text(
        self,
        text: str,
        progress_callback: ProgressCallback | None = None,
        item_callback: ItemCallback | None = None,
    ) -> AnalysisResult:
        self._emit_progress(progress_callback, 'identify', 'Identifying resale items from text')
        raw_items = self._run_json_query(IDENTIFY_TEXT_PROMPT.format(text=text))
        normalized_items = self._normalize_items(raw_items)
        found_count = len(normalized_items)
        self._emit_progress(
            progress_callback,
            'identify',
            (
                f'Found {found_count} candidate '
                f'item{self._pluralize(found_count)} from text input'
            ),
        )
        items: list[PrioritizedItem] = []
        for entry in normalized_items:
            prioritized = self._build_prioritized_item(
                entry,
                progress_callback=progress_callback,
            )
            items.append(prioritized)
            self._emit_item(item_callback, prioritized)
        items.sort(key=lambda row: row.priority_score, reverse=True)
        self._emit_progress(
            progress_callback,
            'rank',
            f'Ranked {len(items)} item{self._pluralize(len(items))} by resale priority',
        )
        return AnalysisResult(
            items=items,
            provider=self.settings.provider_mode,
            model=self.settings.model,
        )

    def analyze_images(
        self,
        image_paths: list[Path],
        progress_callback: ProgressCallback | None = None,
        item_callback: ItemCallback | None = None,
    ) -> AnalysisResult:
        prioritized: list[PrioritizedItem] = []
        warnings: list[str] = []
        image_count = len(image_paths)
        self._emit_progress(
            progress_callback,
            'identify',
            (
                f'Analyzing {image_count} '
                f'image{self._pluralize(image_count)} for distinct items'
            ),
        )
        for image_index, image_path in enumerate(image_paths, start=1):
            try:
                raw_items = self._run_json_query(
                    IDENTIFY_IMAGE_PROMPT,
                    image_path=image_path,
                )
            except Exception as exc:
                warnings.append(f"{image_path.name}: {exc}")
                continue
            normalized_items = self._normalize_items(raw_items)
            if self._should_retry_as_itemized(normalized_items):
                self._emit_progress(
                    progress_callback,
                    'identify',
                    (
                        f'Image {image_index} of {len(image_paths)} returned a grouped lot; '
                        'retrying for individual items'
                    ),
                )
                normalized_items = self._expand_grouped_image_items(image_path, normalized_items[0])
            media_items = self._extract_media_shelf_items(image_path)
            chosen_items = self._select_image_items(
                generic_items=normalized_items,
                media_items=media_items,
            )
            if media_items and chosen_items == media_items:
                self._emit_progress(
                    progress_callback,
                    'identify',
                    (
                        f'Image {image_index} of {image_count}: '
                        'using shelf-media extraction for books and dvds'
                    ),
                )
            item_count = len(chosen_items)
            self._emit_progress(
                progress_callback,
                'identify',
                (
                    f'Image {image_index} of {image_count}: '
                    f'found {item_count} candidate item{self._pluralize(item_count)}'
                ),
            )
            for entry in chosen_items:
                prioritized_item = self._build_prioritized_item(
                    entry,
                    source_image=image_path.name,
                    progress_callback=progress_callback,
                )
                prioritized.append(prioritized_item)
                self._emit_item(item_callback, prioritized_item)
        merged_items = self._merge_duplicate_items(prioritized)
        merged_count = len(merged_items)
        self._emit_progress(
            progress_callback,
            'rank',
            (
                f'Ranked {merged_count} '
                f'item{self._pluralize(merged_count)} by resale priority'
            ),
        )
        return AnalysisResult(
            items=merged_items,
            provider=self.settings.provider_mode,
            model=self.settings.model,
            warnings=warnings,
        )

    def _build_prioritized_item(
        self,
        payload: dict[str, Any],
        *,
        source_image: str | None = None,
        progress_callback: Callable[[str, str], None] | None = None,
    ) -> PrioritizedItem:
        item = IdentifiedItem.model_validate(self._normalize_identified_item(payload))
        pricing = self._research_price(item, progress_callback=progress_callback)
        prioritized = assign_priority(PrioritizedItem(item=item, pricing=pricing))
        if source_image:
            prioritized.source_images = [source_image]
        return prioritized

    def _research_price(
        self,
        item: IdentifiedItem,
        *,
        progress_callback: Callable[[str, str], None] | None = None,
    ) -> PriceBand:
        self._emit_progress(
            progress_callback,
            'price_research',
            f'Researching market prices for {item.name}',
        )
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
            payload = [payload]
        if isinstance(payload, list):
            normalized: list[dict[str, Any]] = []
            for entry in payload:
                if not isinstance(entry, dict):
                    continue
                name = entry.get('name')
                if not isinstance(name, str) or not name.strip():
                    continue
                normalized.append(entry)
            return normalized
        raise ValueError(
            "Expected JSON object or array from Hermes, "
            f"got: {type(payload).__name__}"
        )

    def _should_retry_as_itemized(self, items: list[dict[str, Any]]) -> bool:
        if len(items) != 1:
            return False
        item = items[0]
        name = str(item.get('name', '')).strip().lower()
        notes = str(item.get('notes', '') or '').strip().lower()
        grouped_markers = (
            'assorted',
            'lot',
            'bundle',
            'collection',
            'shelf',
            'mixed',
            'set',
            'library',
        )
        haystack = f'{name} {notes}'
        return any(marker in haystack for marker in grouped_markers)

    def _expand_grouped_image_items(
        self,
        image_path: Path,
        grouped_item: dict[str, Any],
    ) -> list[dict[str, Any]]:
        payload = self._run_json_query(
            ITEMIZE_IMAGE_PROMPT.format(
                grouped_item_json=json.dumps(grouped_item, ensure_ascii=False)
            ),
            image_path=image_path,
        )
        return self._normalize_items(payload)

    def _extract_media_shelf_items(self, image_path: Path) -> list[dict[str, Any]]:
        payload = self._run_json_query(MEDIA_SHELF_PROMPT, image_path=image_path)
        items = self._normalize_items(payload)
        if items:
            return self._merge_raw_items([], items)
        section_items: list[dict[str, Any]] = []
        for section_name in ('left third', 'center third', 'right third'):
            section_payload = self._run_json_query(
                MEDIA_SHELF_SECTION_PROMPT.format(section_name=section_name),
                image_path=image_path,
            )
            section_items.extend(self._normalize_items(section_payload))
        return self._merge_raw_items([], section_items)

    def _select_image_items(
        self,
        *,
        generic_items: list[dict[str, Any]],
        media_items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not media_items:
            return generic_items
        if len(media_items) >= 2:
            return media_items
        if not generic_items:
            return media_items
        if not any(self._is_media_item_payload(item) for item in generic_items):
            return media_items
        return self._merge_raw_items(generic_items, media_items)

    def _merge_raw_items(
        self,
        generic_items: list[dict[str, Any]],
        media_items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: dict[tuple[str, str], dict[str, Any]] = {}
        for item in [*generic_items, *media_items]:
            key = self._raw_item_dedupe_key(item)
            existing = merged.get(key)
            if existing is None or float(item.get('confidence', 0.0) or 0.0) > float(
                existing.get('confidence', 0.0) or 0.0
            ):
                merged[key] = item
        return list(merged.values())

    @staticmethod
    def _is_media_item_payload(item: dict[str, Any]) -> bool:
        category = str(item.get('category', '')).strip().lower()
        return category in {'book', 'dvd'}

    @staticmethod
    def _raw_item_dedupe_key(item: dict[str, Any]) -> tuple[str, str]:
        return (
            str(item.get('category', '')).strip().lower(),
            SmartProvider._normalize_dedupe_name(str(item.get('name', ''))),
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
            should_prefer_item = item.item.confidence > existing.item.confidence
            existing.item.confidence = max(existing.item.confidence, item.item.confidence)
            if should_prefer_item:
                existing.item.name = item.item.name
                existing.item.category = item.item.category
                existing.item.condition_guess = item.item.condition_guess
                existing.item.identifiers = dict(item.item.identifiers)
                existing.item.notes = item.item.notes
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
            SmartProvider._normalize_dedupe_name(item.item.name),
        )

    @staticmethod
    def _normalize_dedupe_name(name: str) -> str:
        lowered = name.strip().lower()
        normalized = re.sub(r'[^0-9a-z]+', ' ', lowered)
        return ' '.join(normalized.split())

    @staticmethod
    def _emit_progress(
        progress_callback: ProgressCallback | None,
        step: str,
        message: str,
    ) -> None:
        if progress_callback is not None:
            progress_callback(step, message)

    @staticmethod
    def _emit_item(
        item_callback: ItemCallback | None,
        item: PrioritizedItem,
    ) -> None:
        if item_callback is not None:
            item_callback(item)

    @staticmethod
    def _pluralize(count: int) -> str:
        return '' if count == 1 else 's'

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
