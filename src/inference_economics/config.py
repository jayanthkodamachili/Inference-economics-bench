"""Benchmark configuration, validated on construction."""

from __future__ import annotations

from dataclasses import dataclass

from inference_economics.models import MODEL_NAMES
from inference_economics.pricing import DEFAULT_INSTANCE, PRICE_SNAPSHOT_USD_PER_HOUR

MIN_LATENCY_SAMPLES = 50


class ConfigError(ValueError):
    """Raised when a configuration would produce a measurement that cannot be trusted."""


@dataclass(frozen=True)
class BenchConfig:
    """One benchmark run: a corpus, a model set, and how to time them."""

    n_docs: int = 12_000
    vocab_size: int = 1_500
    n_classes: int = 4
    doc_length: int = 48
    phrase_rate: float = 0.35
    label_noise: float = 0.03
    train_share: float = 0.7
    models: tuple[str, ...] = MODEL_NAMES
    batch_sizes: tuple[int, ...] = (1, 8, 64)
    latency_repeats: int = 200
    latency_warmups: int = 20
    instance_type: str = DEFAULT_INSTANCE
    seed: int = 20260219

    def __post_init__(self) -> None:
        if self.n_docs < 500:
            raise ConfigError(
                f"n_docs={self.n_docs} is too small to split into a train and evaluation "
                "set that say anything; accuracy differences would be inside the noise."
            )
        if not 0.0 <= self.phrase_rate <= 1.0:
            raise ConfigError(f"phrase_rate={self.phrase_rate} must be between 0 and 1")
        if not 0.0 <= self.label_noise < 0.5:
            raise ConfigError(
                f"label_noise={self.label_noise} must be in [0, 0.5); at or above a half "
                "the labels carry less information than a coin and no model can be ranked."
            )
        if not 0.1 <= self.train_share <= 0.9:
            raise ConfigError(f"train_share={self.train_share} must be between 0.1 and 0.9")
        unknown = [m for m in self.models if m not in MODEL_NAMES]
        if unknown:
            raise ConfigError(f"unknown models {unknown}, expected from {list(MODEL_NAMES)}")
        if not self.models:
            raise ConfigError("at least one model is required")
        if any(b < 1 for b in self.batch_sizes):
            raise ConfigError(f"batch sizes must be at least 1, got {self.batch_sizes}")
        if self.latency_repeats < MIN_LATENCY_SAMPLES:
            raise ConfigError(
                f"latency_repeats={self.latency_repeats} is below {MIN_LATENCY_SAMPLES}. "
                "A p99 estimated from fewer samples is the maximum with extra steps, and "
                "the tail is the number this benchmark publishes."
            )
        if self.latency_warmups < 1:
            raise ConfigError(
                "at least one warmup is required; the first call into these models is an "
                "order of magnitude slower than the steady state and would be published "
                "as a tail no request ever experiences."
            )
        if self.instance_type not in PRICE_SNAPSHOT_USD_PER_HOUR:
            raise ConfigError(
                f"no snapshot price for {self.instance_type!r}; known types are "
                f"{sorted(PRICE_SNAPSHOT_USD_PER_HOUR)}"
            )

    def as_dict(self) -> dict:
        return {
            "n_docs": self.n_docs, "vocab_size": self.vocab_size,
            "n_classes": self.n_classes, "doc_length": self.doc_length,
            "phrase_rate": self.phrase_rate, "label_noise": self.label_noise,
            "train_share": self.train_share, "models": list(self.models),
            "batch_sizes": list(self.batch_sizes),
            "latency_repeats": self.latency_repeats,
            "latency_warmups": self.latency_warmups,
            "instance_type": self.instance_type, "seed": self.seed,
        }
