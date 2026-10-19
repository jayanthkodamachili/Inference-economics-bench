# ADR-004: Costs come from a committed snapshot, with a live path beside it

Status: accepted

## Context

Every cost in this repository is a latency measurement multiplied by a machine price. The
latency is mine. The price belongs to a vendor and changes without warning.

## Decision

A static table of on demand prices, with the region and the date it was taken, committed to
version control. Every published figure uses it. A second function queries the AWS Price
List API through boto3 and is not used for anything published.

## Why both

The snapshot is for reproducibility. A reader who reruns this next year should get the same
costs from the same latencies, and a figure that changed because a vendor changed a price
while nothing in the repository changed is a figure nobody can check. The date is committed
with the numbers so staleness is visible rather than assumed away.

The live path is for the question an operator actually has, which is what this costs now,
and it keeps the cost model from quietly rotting into a historical artifact.

## How the live path behaves

It falls back to the snapshot on any failure, and the returned `Price` records which source
it came from. Every table prints that source. A benchmark that cannot run because a pricing
endpoint is unreachable has turned a reporting detail into a hard dependency; a benchmark
that silently substitutes one number for another is worse. Recording the provenance is what
makes the fallback safe rather than sloppy.

Callers who genuinely need a live price can pass `fall_back=False` and get an exception.

## Testing

The live path is covered by tests that stub the botocore client, not by calling AWS. A test
that hits a vendor API needs credentials, costs money eventually, fails when the network
does, and tests the vendor rather than this code. The stubbed tests cover the response
shape, which is the part that is actually easy to get wrong: `terms.OnDemand` is a dict
keyed by an offer code, containing `priceDimensions` keyed by a rate code, containing
`pricePerUnit.USD` as a string, and none of those keys are stable.

## Consequence

Costs are compute only: no storage, egress, idle capacity or engineer time. Every one of
those falls harder on the expensive models, so the published gaps are lower bounds. This is
stated in the README and in ADR-005 rather than left for a reader to work out.
