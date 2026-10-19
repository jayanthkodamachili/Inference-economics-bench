# ADR-001: The bigram model is in the comparison on purpose

Status: accepted

## Context

The task generator mixes unigram signal with ordered word pairs that invert the class. The
second kind is what gives an expensive model something to find that a simple one cannot.

The obvious comparison is then a transformer against a unigram linear model, and the
transformer wins. That comparison is rigged, and it is rigged in the direction the
literature already leans.

## Decision

Include TF-IDF with bigrams and logistic regression as a first class model, and treat it
as the default competitor whenever order dependent signal is involved.

## Why

Because it is the honest alternative. A unigram model cannot represent word order at all,
so "the neural model captures word order and the linear model does not" is a statement
about the vectoriser, not about the models. Changing `ngram_range=(1, 1)` to `(1, 2)` is
one argument, costs almost nothing per call, and recovers most of the available signal.

The experiment bears this out. At phrase rate 0.8 the bigram model reaches 0.9717 against
the transformer's 0.9517, so the cheap model is not merely competitive there, it is ahead.
A comparison without it would have shown the transformer winning and attributed the win to
the architecture rather than to the feature space.

The same reasoning put SVD before gradient boosting rather than boosting raw sparse
TF-IDF. Boosting on a high dimensional sparse matrix is slow enough that nobody deploys
it, so including it unreduced would have produced a second strawman, this time in the
opposite direction.

## Consequence

Every model in the set has a plausible reason to be deployed, and no result here depends
on a competitor being handicapped. It also means the headline is less tidy than it would
otherwise be: the most accurate model is gradient boosting, not the transformer, and the
transformer is beaten by logistic regression entirely when the task has no order dependent
signal.
