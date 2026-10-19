"""Command line entry point.

Exit code 4 is the one worth having: the model you named is dominated, meaning another
model in the comparison is both more accurate and cheaper per call. There is no budget at
which the dominated one is the right answer, and no accuracy table alone can tell you that.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from inference_economics import economics
from inference_economics.config import BenchConfig, ConfigError
from inference_economics.models import MODEL_NAMES
from inference_economics.pricing import (
    PRICE_SNAPSHOT_USD_PER_HOUR,
    from_aws,
    from_snapshot,
)
from inference_economics.report import Report, run

EXIT_ON_FRONTIER = 0
EXIT_OVERPAYING = 1
EXIT_BAD_CONFIG = 2
EXIT_INTERNAL_ERROR = 3
EXIT_DOMINATED = 4
EXIT_NOTHING_MEETS_SLA = 5

NEGLIGIBLE_ACCURACY_POINTS = 1.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inference-economics",
        description="Price a model's accuracy against what it costs to serve.",
    )
    parser.add_argument("--choose", choices=MODEL_NAMES, default="transformer",
                        help="the model you are proposing to deploy")
    parser.add_argument("--n-docs", type=int, default=12_000)
    parser.add_argument("--phrase-rate", type=float, default=0.35)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--latency-repeats", type=int, default=200)
    parser.add_argument("--instance-type", default="c7i.large",
                        choices=sorted(PRICE_SNAPSHOT_USD_PER_HOUR))
    parser.add_argument("--p99-budget-ms", type=float, default=None,
                        help="tail latency the service must meet")
    parser.add_argument("--live-pricing", action="store_true",
                        help="query the AWS Price List API instead of the snapshot")
    parser.add_argument("--json", action="store_true")
    return parser


def classify(report: Report, chosen: str, batch_size: int,
             p99_budget_ms: float | None) -> int:
    points = report.points(batch_size)
    by_name = {p.model: p for p in points}
    choice = by_name[chosen]

    if p99_budget_ms is not None:
        if economics.best_within_sla(points, p99_budget_ms) is None:
            return EXIT_NOTHING_MEETS_SLA
        if choice.p99_ms > p99_budget_ms:
            return EXIT_OVERPAYING

    if any(economics.dominates(other, choice) for other in points if other is not choice):
        return EXIT_DOMINATED

    cheaper_and_close = [
        p for p in points
        if p.usd_per_million < choice.usd_per_million
        and (choice.accuracy - p.accuracy) * 100 < NEGLIGIBLE_ACCURACY_POINTS
    ]
    if cheaper_and_close:
        return EXIT_OVERPAYING
    return EXIT_ON_FRONTIER


def render(report: Report, chosen: str, batch_size: int,
           p99_budget_ms: float | None) -> str:
    points = report.points(batch_size)
    by_name = {p.model: p for p in points}
    choice = by_name[chosen]
    price = report.price

    lines = [
        f"instance {price.instance_type} at ${price.usd_per_hour:.5f}/hour "
        f"({price.source}, {price.as_of}), batch size {batch_size}",
        "",
        f"{'model':<17}{'accuracy':>10}{'per item ms':>13}{'p99 ms':>9}"
        f"{'$ / million':>13}",
        "-" * 62,
    ]
    for point in sorted(points, key=lambda p: p.usd_per_million):
        marker = "  <- chosen" if point.model == chosen else ""
        lines.append(
            f"{point.model:<17}{point.accuracy:>10.4f}{point.per_item_ms:>13.4f}"
            f"{point.p99_ms:>9.3f}{point.usd_per_million:>13.4f}{marker}"
        )
    lines.append("-" * 62)

    on_frontier = [p.model for p in economics.frontier(points)]
    lines.append("on the frontier: " + ", ".join(on_frontier))
    for beaten, by in economics.dominated(points):
        lines.append(
            f"dominated: {beaten.model} is beaten by {by.model}, which is more accurate "
            f"and cheaper per call"
        )

    better = [p for p in points if p.accuracy > choice.accuracy]
    if better:
        nearest = min(better, key=lambda p: p.usd_per_million)
        rate = economics.accuracy_per_extra_dollar(choice, nearest)
        lines.append(
            f"upgrading from {chosen} to {nearest.model} buys "
            f"{(nearest.accuracy - choice.accuracy) * 100:.2f} accuracy points at "
            f"{rate:.2f} points per extra dollar per million calls"
        )

    if p99_budget_ms is not None:
        fits = economics.best_within_sla(points, p99_budget_ms)
        if fits is None:
            lines.append(f"no model meets a p99 budget of {p99_budget_ms:.2f} ms")
        else:
            lines.append(
                f"most accurate model inside a {p99_budget_ms:.2f} ms p99 budget: "
                f"{fits.model} at {fits.accuracy:.4f}"
            )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = BenchConfig(
            n_docs=args.n_docs, phrase_rate=args.phrase_rate,
            batch_sizes=(args.batch_size,), latency_repeats=args.latency_repeats,
            instance_type=args.instance_type,
        )
    except ConfigError as exc:
        print(f"invalid benchmark: {exc}", file=sys.stderr)
        return EXIT_BAD_CONFIG

    try:
        price = (from_aws(args.instance_type) if args.live_pricing
                 else from_snapshot(args.instance_type))
        report = run(config, price)
    except Exception as exc:  # noqa: BLE001
        print(f"benchmark failed to complete: {exc}", file=sys.stderr)
        return EXIT_INTERNAL_ERROR

    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        print(render(report, args.choose, args.batch_size, args.p99_budget_ms))
    return classify(report, args.choose, args.batch_size, args.p99_budget_ms)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
