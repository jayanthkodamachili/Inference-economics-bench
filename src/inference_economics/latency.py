"""Measure what one prediction costs, in the two shapes serving actually takes.

Single request latency and batched throughput are different measurements and they rank
models differently, which is the reason both are here. A model with a large fixed cost per
call looks terrible on the first and can look fine on the second, and a team that measured
only one of them will be surprised by the other in production.

Percentiles rather than means. A mean latency is a number almost no request experiences,
and the number in an SLA is nearly always a tail one.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LatencyProfile:
    """Per request cost for one model at one batch size."""

    model: str
    batch_size: int
    samples: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    per_item_ms: float

    @property
    def items_per_second(self) -> float:
        return 1000.0 / self.per_item_ms if self.per_item_ms > 0 else float("inf")

    def as_dict(self) -> dict:
        return {
            "model": self.model, "batch_size": self.batch_size, "samples": self.samples,
            "p50_ms": round(self.p50_ms, 4), "p95_ms": round(self.p95_ms, 4),
            "p99_ms": round(self.p99_ms, 4), "mean_ms": round(self.mean_ms, 4),
            "per_item_ms": round(self.per_item_ms, 5),
            "items_per_second": round(self.items_per_second, 1),
        }


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.array(values), q))


def measure(model, documents: list[str], batch_size: int = 1, repeats: int = 200,
            warmups: int = 20) -> LatencyProfile:
    """Time repeated calls, cycling through the documents.

    Warmups are discarded. The first call into any of these models is unrepresentative:
    sklearn resolves dispatch, torch allocates workspace, and the page cache is cold. The
    first measurement of the transformer was roughly an order of magnitude above its
    median, and including it would have published a tail that no steady state request
    would ever see.

    Documents are cycled rather than reused, so repeated calls do not measure one
    document that happens to be short.
    """
    total_calls = repeats + warmups
    calls = []
    if batch_size == 1:
        # The document is bound as a default argument rather than captured by closure.
        # Captured by closure, every lambda would see the final value of the loop
        # variable and the benchmark would time one document repeatedly.
        for index in range(total_calls):
            chosen = documents[index % len(documents)]
            calls.append(lambda document=chosen: model.predict_one(document))
    else:
        span = max(1, len(documents) - batch_size)
        for index in range(total_calls):
            start = (index * batch_size) % span
            calls.append(
                lambda s=start: model.predict_batch(documents[s : s + batch_size])
            )

    for call in calls[:warmups]:
        call()

    timings = []
    for call in calls[warmups:]:
        started = time.perf_counter()
        call()
        timings.append((time.perf_counter() - started) * 1000.0)

    return LatencyProfile(
        model=model.name, batch_size=batch_size, samples=len(timings),
        p50_ms=_percentile(timings, 50), p95_ms=_percentile(timings, 95),
        p99_ms=_percentile(timings, 99), mean_ms=statistics.fmean(timings),
        per_item_ms=_percentile(timings, 50) / batch_size,
    )
