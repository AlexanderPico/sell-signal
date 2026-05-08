from __future__ import annotations

from pydantic import BaseModel, Field


class PriceBand(BaseModel):
    used_low: float | None = None
    used_high: float | None = None
    used_median: float | None = None
    new_low: float | None = None
    new_high: float | None = None
    new_median: float | None = None
    currency: str = "USD"
    evidence: list[str] = Field(default_factory=list)


class IdentifiedItem(BaseModel):
    name: str
    category: str
    confidence: float = 0.0
    condition_guess: str | None = None
    identifiers: dict[str, str] = Field(default_factory=dict)
    notes: str | None = None


class PrioritizedItem(BaseModel):
    item: IdentifiedItem
    pricing: PriceBand = Field(default_factory=PriceBand)
    priority_score: float = 0.0
    priority_label: str = "inspect"
    why: list[str] = Field(default_factory=list)
    source_images: list[str] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    items: list[PrioritizedItem] = Field(default_factory=list)
    provider: str
    model: str
    warnings: list[str] = Field(default_factory=list)
