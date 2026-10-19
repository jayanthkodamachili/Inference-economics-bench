"""E4: how much order dependent signal does there have to be before the expensive model
pays for itself?

The corpus generator mixes two kinds of signal. Unigram signal that any linear model over
word counts can find, and ordered word pairs that invert the class, which a unigram model
cannot represent at all. phrase_rate is the share of documents carrying the second kind.

At phrase_rate 0 the task is pure bag of words, and there should be nothing for an
expensive model to find that a cheap one cannot. As it rises, the cheap unigram model
should fall behind and the question becomes whether the gap is worth what it costs.

The answer is a number, and it is the number nobody computes before choosing a model.
"""

from __future__ import annotations

from _common import table, write

from inference_economics import economics
from inference_economics.config import BenchConfig
from inference_economics.report import run

PHRASE_RATES = (0.0, 0.2, 0.4, 0.6, 0.8)
N_DOCS = 8_000
CHEAP = "logreg_unigram"


def main() -> int:
    print(__doc__)
    print(f"Five corpora at {N_DOCS:,} documents, all five models on each.\n")

    rows, cells = [], []
    for rate in PHRASE_RATES:
        config = BenchConfig(n_docs=N_DOCS, phrase_rate=rate, batch_sizes=(1,),
                             latency_repeats=100)
        report = run(config)
        points = {p.model: p for p in report.points(1)}
        for p in points.values():
            cells.append({"phrase_rate": rate, **p.as_dict()})

        cheap = points[CHEAP]
        best = max(points.values(), key=lambda p: p.accuracy)
        gap = (best.accuracy - cheap.accuracy) * 100
        value = economics.accuracy_per_extra_dollar(cheap, best)
        rows.append({
            "phrase_rate": rate,
            "cheap_accuracy": round(cheap.accuracy, 4),
            "best_model": best.model,
            "best_accuracy": round(best.accuracy, 4),
            "gap_points": round(gap, 2),
            "extra_usd_per_million": round(best.usd_per_million - cheap.usd_per_million, 4),
            "points_per_extra_dollar": round(value, 1) if value != float("inf") else None,
            "bigram_accuracy": round(points["logreg_ngram2"].accuracy, 4),
            "transformer_accuracy": round(points["transformer"].accuracy, 4),
        })
        print(f"  phrase_rate {rate:.1f}: {CHEAP} {cheap.accuracy:.4f}, "
              f"best {best.model} {best.accuracy:.4f}, gap {gap:.2f} points")

    print("\naccuracy by model as order dependent signal increases\n")
    table(["phrase rate", "unigram", "bigram", "transformer", "best model", "gap pts"],
          [[f"{r['phrase_rate']:.1f}", f"{r['cheap_accuracy']:.4f}",
            f"{r['bigram_accuracy']:.4f}", f"{r['transformer_accuracy']:.4f}",
            r["best_model"], f"{r['gap_points']:.2f}"] for r in rows],
          [13, 10, 10, 13, 16, 9])

    print("\nwhat the upgrade from the cheapest useful model costs\n")
    table(["phrase rate", "extra $/M", "points", "points per $"],
          [[f"{r['phrase_rate']:.1f}", f"{r['extra_usd_per_million']:.4f}",
            f"{r['gap_points']:.2f}",
            "free" if r["points_per_extra_dollar"] is None
            else f"{r['points_per_extra_dollar']:.1f}"] for r in rows],
          [13, 12, 9, 14])

    bigram_lift = [round((r["bigram_accuracy"] - r["cheap_accuracy"]) * 100, 2)
                   for r in rows]

    print(f"\nAt phrase_rate 0.0 the bigram model gains "
          f"{bigram_lift[0]:.2f} accuracy points over the unigram model, which is the "
          "control: with no order dependent signal there is nothing for it to find.")
    print(f"At phrase_rate {rows[-1]['phrase_rate']:.1f} it gains {bigram_lift[-1]:.2f} "
          "points, and that lift costs almost nothing, because a second ngram range is "
          "one argument to a vectoriser.")

    best_models = {r["best_model"] for r in rows}
    print(f"\nThe most accurate model across the sweep is "
          f"{'always ' + best_models.pop() if len(best_models) == 1 else 'not constant'}.")

    print(f"\nThe gap the expensive model has to justify ranges from "
          f"{min(r['gap_points'] for r in rows):.2f} to "
          f"{max(r['gap_points'] for r in rows):.2f} accuracy points across the sweep, so "
          "the same upgrade is worth very different amounts depending on a property of "
          "the data that nobody measures before choosing a model.")

    write("e4_when_does_it_pay", {
        "experiment": "e4_when_does_it_pay",
        "question": "How much order dependent signal justifies the expensive model?",
        "n_docs": N_DOCS, "phrase_rates": list(PHRASE_RATES),
        "cheap_model": CHEAP, "rows": rows, "cells": cells,
        "bigram_lift_points": bigram_lift,
        "bigram_lift_at_zero": bigram_lift[0],
        "bigram_lift_at_max": bigram_lift[-1],
        "min_gap_points": min(r["gap_points"] for r in rows),
        "max_gap_points": max(r["gap_points"] for r in rows),
        "best_models": sorted({r["best_model"] for r in rows}),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
