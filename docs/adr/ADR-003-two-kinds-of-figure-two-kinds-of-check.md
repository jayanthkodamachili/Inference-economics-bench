# ADR-003: Accuracy is re-measured, latency is recorded

Status: accepted, after nearly publishing a false finding

## Context

This repository publishes two kinds of number and they behave differently.

Accuracy is deterministic. Same seed, same corpus, same split, same model, same figure to
the last digit.

Latency is not. The same model timed five times varies by up to 1.17 on this machine, and
between separate runs the spread is larger still: gradient boosting priced at 0.5177 per
million in one run and 0.7054 in another.

## What went wrong first

E2 asks whether batching changes the cost ordering of the models. The first run said the
ordering was identical at every batch size. The second run said it changed. Both runs were
correct and the conclusion was not, because the two models that swapped places differ by
less than the harness can resolve, and which one measures cheaper is decided by noise.

Either answer could have been published. The second is the more interesting one, which is
exactly why it was the more dangerous.

## Decision

Three things, together.

**The experiment measures its own precision.** E2 times each model five separate times,
publishes the spread, and uses the worst of them as the resolution of the instrument.

**Models inside that resolution are tied, not ranked.** A reordering is reported only when
it is larger than the instrument, and a tie that resolves at a larger batch size is
reported as a resolution rather than a reversal. Those are different claims and the
experiment now distinguishes them.

**The two kinds of figure are checked differently.** `tools/check_reproducible.py` re-fits
the cheap models on every verify and asserts their recorded accuracies hold exactly.
Everything timed is read from the recorded run in `docs/experiments/` and regenerated only
by `make bench`.

## Why not re-measure the timings in verify

Because a check that re-measures a non-deterministic quantity and demands it match the
published figure fails at random, and a check that fails at random is one everybody learns
to ignore. Once ignored it is no longer protecting the deterministic figures either, which
is the real loss.

Recording the run also makes the benchmark reviewable: `docs/experiments/` is in version
control, so rerunning produces a diff showing exactly which published figures moved.

## Consequence

The transformer's accuracy is not re-measured on every verify, because retraining it costs
most of two minutes and a check nobody runs protects nothing. It is covered by the recorded
run and by the slow tests, which is a weaker guarantee, and the README says so rather than
implying every number is re-derived.
