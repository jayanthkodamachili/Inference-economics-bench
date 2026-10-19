"""E2: does the ranking survive batching?

E1 timed single request serving, where every fixed per call cost is paid once per
prediction. Most inference is not served that way. Queue a few hundred milliseconds of
requests, run them together, and the fixed costs are divided by the batch size.

This experiment was called e2_batching_rewrites_it until it ran, and the file was renamed
to a question rather than rephrased around what it found.

It also very nearly published a false finding, which is worth recording. The first run
reported an identical cost ordering at every batch size. The second reported that the
ordering changed, which would have been the more interesting result. Both runs were
correct; the two models that swapped places differ in cost by less than this harness can
resolve, and which one measures cheaper is decided by noise.

So the experiment now measures its own precision first, by timing each model several
separate times and recording the spread. Two models whose costs differ by less than that
spread are reported as tied rather than ranked, and a reordering is only claimed when it
is larger than the instrument. A benchmark that cannot say how precise it is cannot say
that an ordering changed.
"""

from __future__ import annotations

from _common import table, write

from inference_economics import economics
from inference_economics.config import BenchConfig
from inference_economics.corpus import generate, split
from inference_economics.report import run

_TRAIN = None
_EVAL_TEXT: list[str] = []

BATCHES = (1, 8, 64)
CONFIG = BenchConfig(n_docs=12_000, phrase_rate=0.35, batch_sizes=BATCHES,
                     latency_repeats=120)


PRECISION_REPEATS = 5


def _precision(report, documents) -> dict[str, float]:
    """Time each model several more times and report the spread of per item cost.

    Separate measure() calls on an already fitted model, so this isolates timing noise
    from anything about training. The ratio of the largest to the smallest is used as the
    resolution of the instrument: differences smaller than it are not differences.
    """
    from inference_economics.latency import measure
    from inference_economics.models import build

    spreads = {}
    for result in report.results:
        model = build(result.name)
        model.fit(_TRAIN)
        costs = [
            measure(model, documents, batch_size=1, repeats=CONFIG.latency_repeats,
                    warmups=CONFIG.latency_warmups).per_item_ms
            for _ in range(PRECISION_REPEATS)
        ]
        spreads[result.name] = max(costs) / min(costs) if min(costs) > 0 else 1.0
    return spreads


def _tied(a: float, b: float, tolerance: float) -> bool:
    return max(a, b) / min(a, b) <= tolerance if min(a, b) > 0 else True


def _ordering_with_ties(points, tolerance: float) -> tuple:
    """Cheapest first, with models inside the resolution grouped together."""
    ordered = sorted(points, key=lambda p: p.usd_per_million)
    groups, current = [], [ordered[0]]
    for point in ordered[1:]:
        if _tied(current[-1].usd_per_million, point.usd_per_million, tolerance):
            current.append(point)
        else:
            groups.append(current)
            current = [point]
    groups.append(current)
    return tuple(tuple(sorted(p.model for p in g)) for g in groups)


def _contradictions(per_batch, batches, tolerance) -> list[str]:
    """Pairs whose order genuinely reverses between batch sizes.

    A tie that resolves is not a reversal. If two models are indistinguishable at batch 1
    and separate at batch 64, the larger batch has resolved something the smaller one
    could not see, and nothing has been reordered. A reversal means one model is clearly
    cheaper at one batch size and the other is clearly cheaper at another, which is the
    only version of this that would invalidate a single batch benchmark.
    """
    names = [p.model for p in per_batch[batches[0]]]
    found = []
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            verdicts = set()
            for batch in batches:
                costs = {p.model: p.usd_per_million for p in per_batch[batch]}
                a, b = costs[first], costs[second]
                if _tied(a, b, tolerance):
                    continue
                verdicts.add(first if a < b else second)
            if len(verdicts) > 1:
                found.append(f"{first} vs {second}")
    return found


def main() -> int:
    print(__doc__)
    report = run(CONFIG)
    price = report.price

    global _TRAIN, _EVAL_TEXT
    corpus = generate(n_docs=CONFIG.n_docs, vocab_size=CONFIG.vocab_size,
                      n_classes=CONFIG.n_classes, doc_length=CONFIG.doc_length,
                      phrase_rate=CONFIG.phrase_rate, label_noise=CONFIG.label_noise,
                      seed=CONFIG.seed)
    _TRAIN, evaluation = split(corpus, CONFIG.train_share, CONFIG.seed)
    _EVAL_TEXT = evaluation.as_text()

    cells, per_batch = [], {}
    for batch in BATCHES:
        points = report.points(batch)
        per_batch[batch] = points
        for p in points:
            cells.append({"batch_size": batch, **p.as_dict()})

    print(f"\ncost per million predictions, {price.instance_type} at "
          f"${price.usd_per_hour:.5f}/hour\n")
    models = [r.name for r in report.results]
    table(["model"] + [f"batch {b}" for b in BATCHES],
          [[m] + [f"{next(p for p in per_batch[b] if p.model == m).usd_per_million:.4f}"
                  for b in BATCHES] for m in models],
          [18] + [12] * len(BATCHES))

    print("\nhow much batching saves, as a multiple of the batch 1 cost\n")
    savings = {}
    for model in models:
        one = next(p for p in per_batch[1] if p.model == model).usd_per_million
        big = next(p for p in per_batch[BATCHES[-1]] if p.model == model).usd_per_million
        savings[model] = one / big if big > 0 else float("inf")
    def _saving(model: str) -> str:
        value = savings[model]
        return "free either way" if value == float("inf") else f"{value:.1f}x cheaper"

    table(["model", f"batch 1 to {BATCHES[-1]}"],
          [[m, _saving(m)] for m in models], [18, 20])

    print(f"\nhow precise is this? timing each model {PRECISION_REPEATS} more times at "
          "batch 1\n")
    spreads = _precision(report, _EVAL_TEXT)
    tolerance = max(spreads.values())
    table(["model", "spread across repeats"],
          [[m, f"x{spreads[m]:.2f}"] for m in models], [18, 24])
    print(f"\nThe worst spread is x{tolerance:.2f}, so two models whose costs differ by "
          f"less than that are reported as tied rather than ranked.")

    orderings = {b: _ordering_with_ties(per_batch[b], tolerance) for b in BATCHES}
    distinct = len(set(orderings.values()))

    print("\ncost ordering, cheapest first, with ties grouped\n")
    for batch in BATCHES:
        rendered = " < ".join("=".join(group) for group in orderings[batch])
        print(f"  batch {batch:>3}: {rendered}")

    paying = {}
    for batch in BATCHES:
        points = per_batch[batch]
        best = max(points, key=lambda p: p.accuracy)
        cheap = next(p for p in points if p.model == "logreg_unigram")
        paying[batch] = best.usd_per_million / cheap.usd_per_million

    print(f"\nThe premium for the most accurate model over the cheapest useful one falls "
          f"from {paying[1]:.1f}x at batch 1 to {paying[BATCHES[-1]]:.1f}x at batch "
          f"{BATCHES[-1]}.")
    reversals = _contradictions(per_batch, BATCHES, tolerance)
    if reversals:
        print(f"\n{len(reversals)} pair(s) genuinely reverse between batch sizes: "
              f"{', '.join(reversals)}. A benchmark at one batch size does not rank "
              "these models.")
    elif distinct > 1:
        print(f"\nThe grouped ordering differs across batch sizes, in {distinct} "
              "arrangements, but no pair reverses. What changes is resolution: models "
              "that are indistinguishable at batch 1 separate at batch 64, because "
              "amortising the fixed cost shrinks the noise relative to the difference. "
              "The larger batch is not a different ranking, it is a sharper one.")
    else:
        print("\nOnce ties are grouped the ordering does not change at all. Batching "
              "moves the size of the gaps, not the shape of the decision.")

    frontier_changes = {b: [p.model for p in economics.frontier(per_batch[b])]
                        for b in BATCHES}
    print(f"\nfrontier at batch 1:  {', '.join(frontier_changes[1])}")
    print(f"frontier at batch {BATCHES[-1]}: {', '.join(frontier_changes[BATCHES[-1]])}")

    write("e2_does_batching_change_the_answer", {
        "experiment": "e2_does_batching_change_the_answer",
        "question": "Does the cost ranking survive batching?",
        "config": CONFIG.as_dict(), "price": price.as_dict(),
        "batches": list(BATCHES), "cells": cells,
        "savings": {m: round(v, 1) if v != float("inf") else None
                    for m, v in savings.items()},
        "orderings": {str(b): [list(g) for g in o] for b, o in orderings.items()},
        "distinct_orderings": distinct,
        "precision_repeats": PRECISION_REPEATS,
        "measurement_spreads": {m: round(v, 3) for m, v in spreads.items()},
        "tie_tolerance": round(tolerance, 3),
        "reversed_pairs": reversals,
        "reversal_count": len(reversals),
        "premium_at_batch_1": round(paying[1], 1),
        "premium_at_largest_batch": round(paying[BATCHES[-1]], 1),
        "largest_batch": BATCHES[-1],
        "frontier_by_batch": {str(b): v for b, v in frontier_changes.items()},
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
