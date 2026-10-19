# ADR-005: What this repository cannot measure

Status: accepted

## Context

The argument here is that accuracy and cost rank models differently. That argument is only
as good as the models and the task behind it, and both have real limits.

## The limits

**No pretrained weights.** The environment this was built in could not reach a model hub,
so the transformer is trained from scratch on the task rather than fine tuned from a
checkpoint. This is the largest limitation. A fine tuned DistilBERT would be more accurate
than anything here and several times more expensive to serve, which would move every number
and strengthen rather than weaken the argument. But it was not measured, so it is not
claimed.

**A synthetic task.** The corpus is generated, which is what makes the order dependent
signal a knob and the control row in E4 possible. It is also why no accuracy here should be
read as predicting accuracy on real text. What transfers is the shape: that a cheap model
captures most of the signal, that the last few points cost disproportionately, and that one
extra ngram range is the control experiment people skip.

**One machine, CPU only.** Two cores, no GPU. A GPU changes the economics of the
transformer more than of any other model here, in its favour on throughput and against it
on idle cost, and none of that is measured. The latency figures are also noisy: the same
model timed five times varies by up to 1.17, which is why models inside that band are
reported as tied.

**Compute only costs.** No storage, no network egress, no idle capacity between requests,
no autoscaling inefficiency, no engineering time to build and maintain the more complex
model. Every one of these falls harder on the expensive models, so the published gaps are
lower bounds on the real difference.

**A dated price snapshot.** Prices are from one region on one date, committed so the
figures reproduce. They will drift. The live AWS path exists for the current number and is
not used for anything published.

**Latency measured in process.** No HTTP server, no serialisation, no network hop, no
cold start, no concurrent load. Real serving adds fixed overhead to every model, which
compresses the relative differences, and adds queueing behaviour under load that none of
this captures.

**Five models, chosen before the results.** They span the range deliberately and were not
changed once the numbers came in. They are not a survey of what is available.

## Consequence

The claim supported is narrow and still worth making: for a given task and machine,
accuracy and cost rank models differently; the difference is large enough to change which
model should be deployed; dominance and budget are computable in an afternoon; and the
cheap control experiment for order dependent signal is one argument to a vectoriser rather
than a neural network.
