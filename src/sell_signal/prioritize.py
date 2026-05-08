from __future__ import annotations

from sell_signal.schema import PrioritizedItem


def assign_priority(item: PrioritizedItem) -> PrioritizedItem:
    median = item.pricing.used_median or item.pricing.new_median or 0.0
    confidence = item.item.confidence
    score = min(100.0, median + confidence * 20)
    reasons: list[str] = []
    if median >= 40:
        reasons.append("meaningful resale value")
    if confidence >= 0.8:
        reasons.append("high identification confidence")
    if not reasons:
        reasons.append("needs manual review")
    label = "sell" if score >= 55 else "inspect" if score >= 20 else "skip"
    item.priority_score = round(score, 1)
    item.priority_label = label
    item.why = reasons
    return item
