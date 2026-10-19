# ADR-002: Every model is charged for the same work

Status: accepted

## Context

The models take different inputs natively. The linear pipelines want strings. The
transformer wants integer tensors of a fixed length with a padding mask.

It would be easy, and common, to hand each model the input it likes best and time what
happens next.

## Decision

Both prediction paths, `predict_one` and `predict_batch`, take strings. Whatever a model
must do to consume a string is part of its cost, including the transformer's tokenisation
and padding.

## Why

Because the alternative measures two different tasks and calls the result a comparison. A
pipeline that receives pre-tokenised tensors has had part of its work done by the
benchmark harness, and in production that work still happens, on the same machine, in the
same request. Excluding it would move the transformer's per call cost in its favour by an
amount nobody reading the table could see.

The same principle is why setup is excluded from the timed repeats but reported separately
rather than hidden: a cost that is real should be visible, and a cost that is amortised
should be shown as amortised rather than silently dropped.

## Consequence

The transformer's measured cost includes parsing, which is a meaningful fraction of it at
batch 1 and almost none of it at batch 64. That difference is part of why the models do
not benefit equally from batching, which is the finding in E2.

A reader who wants the model-only cost can get it by subtracting the majority baseline,
which does nothing except parse. That is what the baseline is for, beyond being an
accuracy floor.
