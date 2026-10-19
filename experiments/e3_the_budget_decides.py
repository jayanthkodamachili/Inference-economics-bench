"""E3: there is no best model, only a best model per budget.

This is an analysis of the measurements E1 already recorded rather than a new benchmark.
Re-timing the same five models to answer a question about the numbers already in hand
would add noise and nothing else, so E3 reads e1_the_frontier.json.

Two sweeps. A cost budget in dollars per million predictions, and a tail latency budget in
milliseconds. They are different constraints and they do not have to select the same model,
which is the point: a team with money and no latency headroom and a team with latency
headroom and no money should deploy different things, and an accuracy table tells both of
them the same thing.
"""

from __future__ import annotations

from _common import read, table, write

from inference_economics import economics

BUDGETS_USD_PER_MILLION = (0.005, 0.01, 0.02, 0.05, 0.20, 1.00)
SLA_P99_MS = (0.5, 1.0, 3.0, 10.0, 100.0)


def main() -> int:
    print(__doc__)
    recorded = read("e1_the_frontier")
    points = [economics.Point(**{k: v for k, v in p.items()}) for p in recorded["points"]]
    price = recorded["price"]

    print(f"Using the measurements recorded by E1 on {price['instance_type']} at "
          f"${price['usd_per_hour']:.5f}/hour.\n")

    budget_rows, budget_choice = [], {}
    for budget in BUDGETS_USD_PER_MILLION:
        chosen = economics.best_within_budget(points, budget)
        budget_choice[budget] = chosen
        budget_rows.append([
            f"${budget:.3f}",
            chosen.model if chosen else "nothing affordable",
            f"{chosen.accuracy:.4f}" if chosen else "-",
        ])
    print("most accurate model inside a cost budget\n")
    table(["$ / million", "model", "accuracy"], budget_rows, [14, 18, 12])

    sla_rows, sla_choice = [], {}
    for sla in SLA_P99_MS:
        chosen = economics.best_within_sla(points, sla)
        sla_choice[sla] = chosen
        sla_rows.append([
            f"{sla:.1f} ms",
            chosen.model if chosen else "nothing fits",
            f"{chosen.accuracy:.4f}" if chosen else "-",
        ])
    print("\nmost accurate model inside a p99 latency budget\n")
    table(["p99 budget", "model", "accuracy"], sla_rows, [14, 18, 12])

    distinct_budget = {c.model for c in budget_choice.values() if c}
    distinct_sla = {c.model for c in sla_choice.values() if c}

    print(f"\nAcross six budgets the answer is {len(distinct_budget)} different models: "
          f"{', '.join(sorted(distinct_budget))}.")
    print(f"Across five latency limits it is {len(distinct_sla)}: "
          f"{', '.join(sorted(distinct_sla))}.")

    disagreements = []
    for budget in BUDGETS_USD_PER_MILLION:
        for sla in SLA_P99_MS:
            by_budget, by_sla = budget_choice[budget], sla_choice[sla]
            if by_budget and by_sla and by_budget.model != by_sla.model:
                disagreements.append((budget, sla, by_budget.model, by_sla.model))

    print(f"\nOf the {len(BUDGETS_USD_PER_MILLION) * len(SLA_P99_MS)} budget and latency "
          f"pairs, {len(disagreements)} select different models depending on which "
          "constraint you read first.")

    # The pair that matters most: where money says one thing and the clock says another.
    upgrades = []
    ordered = sorted(points, key=lambda p: p.usd_per_million)
    for cheaper, dearer in zip(ordered, ordered[1:], strict=False):
        if dearer.accuracy > cheaper.accuracy:
            upgrades.append({
                "from": cheaper.model, "to": dearer.model,
                "points_gained": round((dearer.accuracy - cheaper.accuracy) * 100, 2),
                "extra_usd_per_million": round(
                    dearer.usd_per_million - cheaper.usd_per_million, 4),
                "points_per_extra_dollar": round(
                    economics.accuracy_per_extra_dollar(cheaper, dearer), 2),
            })

    print("\nwhat each step up the frontier buys\n")
    table(["from", "to", "points", "extra $/M", "points per $"],
          [[u["from"], u["to"], f"{u['points_gained']:.2f}",
            f"{u['extra_usd_per_million']:.4f}", f"{u['points_per_extra_dollar']:.1f}"]
           for u in upgrades], [18, 18, 9, 12, 14])

    worst = min(upgrades, key=lambda u: u["points_per_extra_dollar"])
    best = max(upgrades, key=lambda u: u["points_per_extra_dollar"])
    print(f"\nThe best value step is {best['from']} to {best['to']} at "
          f"{best['points_per_extra_dollar']:.1f} accuracy points per extra dollar per "
          "million calls.")
    print(f"The worst is {worst['from']} to {worst['to']} at "
          f"{worst['points_per_extra_dollar']:.1f}, a ratio of "
          f"{best['points_per_extra_dollar'] / worst['points_per_extra_dollar']:.0f} "
          "between the best and worst upgrade on the same frontier.")

    write("e3_the_budget_decides", {
        "experiment": "e3_the_budget_decides",
        "question": "Does one model win, or does the budget pick a different one?",
        "source": "e1_the_frontier", "price": price,
        "budgets": list(BUDGETS_USD_PER_MILLION), "slas_ms": list(SLA_P99_MS),
        "by_budget": {str(b): (c.model if c else None) for b, c in budget_choice.items()},
        "by_sla": {str(s): (c.model if c else None) for s, c in sla_choice.items()},
        "distinct_models_by_budget": len(distinct_budget),
        "distinct_models_by_sla": len(distinct_sla),
        "disagreeing_pairs": len(disagreements),
        "total_pairs": len(BUDGETS_USD_PER_MILLION) * len(SLA_P99_MS),
        "upgrades": upgrades,
        "best_value_step": best, "worst_value_step": worst,
        "value_ratio": round(
            best["points_per_extra_dollar"] / worst["points_per_extra_dollar"], 0),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
