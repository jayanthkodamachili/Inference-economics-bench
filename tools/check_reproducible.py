"""Re-train the cheap models and check the recorded accuracies still hold.

The split this repository runs on: accuracy is deterministic and latency is not. Rather
than let that mean nothing is re-measured, the deterministic half is re-measured here on
every verify.

Only the cheap models are re-fitted. Retraining the transformer costs most of two minutes
and would push this out of the inner loop, and a check nobody runs protects nothing. The
transformer's accuracy is covered by the recorded run and by the slow tests instead, which
is a weaker guarantee and is described as one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

CHEAP_MODELS = ("majority", "logreg_unigram", "logreg_ngram2")
TOLERANCE = 1e-9


def main() -> int:
    import numpy as np

    from inference_economics.config import BenchConfig
    from inference_economics.corpus import generate, split
    from inference_economics.models import build

    recorded_path = ROOT / "docs" / "experiments" / "e1_the_frontier.json"
    if not recorded_path.exists():
        print(f"{recorded_path.name} not found; run make bench first", file=sys.stderr)
        return 2

    recorded = json.loads(recorded_path.read_text())
    config = BenchConfig(**{k: (tuple(v) if isinstance(v, list) else v)
                            for k, v in recorded["config"].items()})
    expected = {p["model"]: p["accuracy"] for p in recorded["points"]}

    corpus = generate(
        n_docs=config.n_docs, vocab_size=config.vocab_size, n_classes=config.n_classes,
        doc_length=config.doc_length, phrase_rate=config.phrase_rate,
        label_noise=config.label_noise, seed=config.seed,
    )
    train, evaluation = split(corpus, config.train_share, config.seed)
    text = evaluation.as_text()

    problems = []
    for name in CHEAP_MODELS:
        if name not in expected:
            continue
        model = build(name)
        model.fit(train)
        accuracy = round(float(np.mean(model.predict_batch(text) == evaluation.labels)), 4)
        if abs(accuracy - expected[name]) > TOLERANCE:
            problems.append(
                f"{name}: recorded {expected[name]:.4f}, re-measured {accuracy:.4f}"
            )
        else:
            print(f"  {name:<18} {accuracy:.4f} matches the recorded run")

    if problems:
        print("\nrecorded accuracies no longer reproduce:\n", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print("\nThe corpus or a model changed. Re-run make bench deliberately, and "
              "check the diff in docs/experiments before committing it.", file=sys.stderr)
        return 1

    print(f"\n{len(CHEAP_MODELS)} recorded accuracies reproduced exactly. The "
          "transformer and every latency figure are from the recorded run and are not "
          "re-measured here; see ADR-003.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
