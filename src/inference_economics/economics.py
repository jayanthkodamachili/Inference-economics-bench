"""Turn accuracy and latency into the comparison people actually have to make.

A table of accuracies ranks models on one axis and hides the other. The operations that
matter here are cheap to define and almost never done: convert latency into money, find
which models are dominated, and answer "which model, given this budget" rather than "which
model".
"""

from __future__ import annotations

from dataclasses import dataclass

SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class Point:
    """One model, priced. Accuracy on one axis, dollars per million calls on the other."""

    model: str
    accuracy: float
    per_item_ms: float
    p99_ms: float
    usd_per_million: float
    size_bytes: int

    def as_dict(self) -> dict:
        return {
            "model": self.model, "accuracy": round(self.accuracy, 4),
            "per_item_ms": round(self.per_item_ms, 5),
            "p99_ms": round(self.p99_ms, 4),
            "usd_per_million": round(self.usd_per_million, 4),
            "size_bytes": self.size_bytes,
        }


def usd_per_million(per_item_ms: float, usd_per_hour: float) -> float:
    """Cost of a million predictions at this per item cost on this machine.

    Compute only. No storage, no network egress, no idle capacity, no autoscaling
    inefficiency, no engineer time. Every one of those makes the expensive model relatively
    worse, so this is a lower bound on the gap, and ADR-005 says so where it matters.
    """
    seconds = per_item_ms / 1000.0
    return seconds * (usd_per_hour / SECONDS_PER_HOUR) * 1_000_000


def dominates(a: Point, b: Point) -> bool:
    """True when a is at least as good as b on both axes, and better on one.

    Cheaper and no less accurate, or more accurate and no more expensive. A model that is
    dominated has no budget at which it is the right answer, which is a stronger statement
    than being worse.
    """
    no_worse = a.accuracy >= b.accuracy and a.usd_per_million <= b.usd_per_million
    strictly_better = a.accuracy > b.accuracy or a.usd_per_million < b.usd_per_million
    return no_worse and strictly_better


def frontier(points: list[Point]) -> list[Point]:
    """The points no other point dominates, cheapest first."""
    kept = [
        p for p in points
        if not any(dominates(other, p) for other in points if other is not p)
    ]
    return sorted(kept, key=lambda p: p.usd_per_million)


def dominated(points: list[Point]) -> list[tuple[Point, Point]]:
    """Each dominated point paired with the cheapest point that dominates it."""
    result = []
    for point in points:
        beaten_by = [o for o in points if o is not point and dominates(o, point)]
        if beaten_by:
            result.append((point, min(beaten_by, key=lambda p: p.usd_per_million)))
    return result


def best_within_budget(points: list[Point], budget_usd_per_million: float) -> Point | None:
    """The most accurate model costing no more than the budget."""
    affordable = [p for p in points if p.usd_per_million <= budget_usd_per_million]
    if not affordable:
        return None
    return max(affordable, key=lambda p: (p.accuracy, -p.usd_per_million))


def best_within_sla(points: list[Point], p99_budget_ms: float) -> Point | None:
    """The most accurate model whose tail latency fits the SLA.

    Tail rather than median, because an SLA written against a median is not an SLA. This
    is a separate question from budget and can select a different model, which is the
    whole reason both exist.
    """
    fast_enough = [p for p in points if p.p99_ms <= p99_budget_ms]
    if not fast_enough:
        return None
    return max(fast_enough, key=lambda p: (p.accuracy, -p.usd_per_million))


def accuracy_per_extra_dollar(cheaper: Point, dearer: Point) -> float:
    """Accuracy points bought per extra dollar per million calls.

    The number that decides an upgrade. Returns infinity when the better model costs no
    more, which is not a degenerate case here: it is what a dominated model looks like
    from the other side.
    """
    extra = dearer.usd_per_million - cheaper.usd_per_million
    gained = (dearer.accuracy - cheaper.accuracy) * 100
    return float("inf") if extra <= 0 else gained / extra
