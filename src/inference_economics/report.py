"""Run the whole benchmark: train, score, time, price, and rank.

Accuracy and cost are produced in one pass on purpose. Measuring them separately is how a
team ends up with an accuracy table from one script and a latency table from another, and
a decision made from whichever was on screen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from inference_economics import economics
from inference_economics.config import BenchConfig
from inference_economics.corpus import generate, split
from inference_economics.latency import LatencyProfile, measure
from inference_economics.models import build
from inference_economics.pricing import Price, from_snapshot


@dataclass(frozen=True)
class ModelResult:
    """Everything measured about one model."""

    name: str
    accuracy: float
    train_seconds: float
    size_bytes: int
    profiles: dict[int, LatencyProfile]

    def point(self, price: Price, batch_size: int = 1) -> economics.Point:
        profile = self.profiles[batch_size]
        return economics.Point(
            model=self.name, accuracy=self.accuracy,
            per_item_ms=profile.per_item_ms, p99_ms=profile.p99_ms,
            usd_per_million=economics.usd_per_million(
                profile.per_item_ms, price.usd_per_hour
            ),
            size_bytes=self.size_bytes,
        )

    def as_dict(self) -> dict:
        return {
            "model": self.name, "accuracy": round(self.accuracy, 4),
            "train_seconds": round(self.train_seconds, 3),
            "size_bytes": self.size_bytes,
            "profiles": {str(k): v.as_dict() for k, v in self.profiles.items()},
        }


@dataclass(frozen=True)
class Report:
    config: BenchConfig
    price: Price
    results: list[ModelResult]

    def points(self, batch_size: int = 1) -> list[economics.Point]:
        return [r.point(self.price, batch_size) for r in self.results]

    def as_dict(self) -> dict:
        return {
            "config": self.config.as_dict(),
            "price": self.price.as_dict(),
            "results": [r.as_dict() for r in self.results],
            "frontier_batch1": [p.model for p in economics.frontier(self.points(1))],
            "dominated_batch1": [
                {"model": d.model, "beaten_by": by.model}
                for d, by in economics.dominated(self.points(1))
            ],
        }


def run(config: BenchConfig, price: Price | None = None) -> Report:
    """Train and measure every configured model on one corpus."""
    import time

    price = price or from_snapshot(config.instance_type)
    corpus = generate(
        n_docs=config.n_docs, vocab_size=config.vocab_size, n_classes=config.n_classes,
        doc_length=config.doc_length, phrase_rate=config.phrase_rate,
        label_noise=config.label_noise, seed=config.seed,
    )
    train, evaluation = split(corpus, config.train_share, config.seed)
    evaluation_text = evaluation.as_text()

    results = []
    for name in config.models:
        model = build(name)
        started = time.perf_counter()
        model.fit(train)
        train_seconds = time.perf_counter() - started

        accuracy = float(np.mean(model.predict_batch(evaluation_text) == evaluation.labels))
        profiles = {
            batch: measure(model, evaluation_text, batch_size=batch,
                           repeats=config.latency_repeats, warmups=config.latency_warmups)
            for batch in config.batch_sizes
        }
        results.append(ModelResult(name, accuracy, train_seconds,
                                   model.size_bytes(), profiles))

    return Report(config=config, price=price, results=results)
