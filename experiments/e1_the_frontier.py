"""E1: the same five models, ranked twice.

Once by accuracy, which is how models are usually compared. Once by accuracy and cost
together, which is the comparison anybody deploying them actually faces.

If the two rankings agree, this repository has no argument to make.
"""

from __future__ import annotations

from _common import table, write

from inference_economics import economics
from inference_economics.config import BenchConfig
from inference_economics.report import run

CONFIG = BenchConfig(n_docs=12_000, phrase_rate=0.35, batch_sizes=(1,),
                     latency_repeats=200)


def main() -> int:
    print(__doc__)
    print("Training five models and timing single request serving.\n")

    report = run(CONFIG)
    points = report.points(1)
    price = report.price

    by_accuracy = sorted(points, key=lambda p: -p.accuracy)
    print("ranked by accuracy alone, the way a model comparison usually ends\n")
    table(["model", "accuracy"],
          [[p.model, f"{p.accuracy:.4f}"] for p in by_accuracy], [18, 12])

    print(f"\nthe same models with cost, on {price.instance_type} at "
          f"${price.usd_per_hour:.5f}/hour ({price.source}, {price.as_of})\n")
    table(["model", "accuracy", "per item ms", "p99 ms", "$ / million", "size KB"],
          [[p.model, f"{p.accuracy:.4f}", f"{p.per_item_ms:.4f}", f"{p.p99_ms:.3f}",
            f"{p.usd_per_million:.4f}", f"{p.size_bytes / 1024:.0f}"]
           for p in sorted(points, key=lambda p: p.usd_per_million)],
          [18, 11, 13, 10, 13, 10])

    on_frontier = economics.frontier(points)
    beaten = economics.dominated(points)

    print(f"\non the frontier: {', '.join(p.model for p in on_frontier)}")
    if beaten:
        for loser, winner in beaten:
            print(f"dominated: {loser.model} is beaten by {winner.model}, which is more "
                  f"accurate ({winner.accuracy:.4f} against {loser.accuracy:.4f}) and "
                  f"cheaper (${winner.usd_per_million:.4f} against "
                  f"${loser.usd_per_million:.4f}) per million calls")
    else:
        print("nothing is dominated: every model is the right answer at some budget")

    best = max(points, key=lambda p: p.accuracy)
    cheapest_useful = min((p for p in points if p.model != "majority"),
                          key=lambda p: p.usd_per_million)
    multiple = best.usd_per_million / cheapest_useful.usd_per_million
    gained = (best.accuracy - cheapest_useful.accuracy) * 100

    print(f"\nThe most accurate model is {best.model} at {best.accuracy:.4f}. The cheapest "
          f"model that does anything is {cheapest_useful.model} at "
          f"{cheapest_useful.accuracy:.4f}.")
    print(f"The accuracy table says the gap is {gained:.2f} points. The cost column says "
          f"the gap is {multiple:.1f} times the price per call.")

    transformer = next((p for p in points if p.model == "transformer"), None)
    if transformer is not None and best.model != "transformer":
        print(f"\nNote which model is on top. {best.model} beats the transformer by "
              f"{(best.accuracy - transformer.accuracy) * 100:.2f} accuracy points while "
              f"costing {best.usd_per_million / transformer.usd_per_million:.1f} times as "
              "much per call. Neither of those is the headline anyone expected.")

    write("e1_the_frontier", {
        "experiment": "e1_the_frontier",
        "question": "Do accuracy and cost rank these models the same way?",
        "config": CONFIG.as_dict(), "price": price.as_dict(),
        "points": [p.as_dict() for p in points],
        "results": [r.as_dict() for r in report.results],
        "frontier": [p.model for p in on_frontier],
        "dominated": [{"model": a.model, "beaten_by": b.model} for a, b in beaten],
        "most_accurate": best.model,
        "most_accurate_accuracy": round(best.accuracy, 4),
        "cheapest_useful": cheapest_useful.model,
        "cheapest_useful_accuracy": round(cheapest_useful.accuracy, 4),
        "accuracy_points_gained": round(gained, 2),
        "cost_multiple": round(multiple, 1),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
