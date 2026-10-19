"""Generate a text classification task deterministically, with a tunable amount of signal
that bag of words cannot see.

The generator exists because the comparison needs a task where an expensive model can
genuinely earn its cost, and where that can be dialled. Two kinds of signal are mixed:

Unigram signal. Each class has words it uses more often. Any linear model over word counts
picks this up, and on a task made only of this a transformer has nothing to add, so it
would be strictly dominated by construction. That would be a strawman rather than a
finding.

Phrase signal. Ordered word pairs that invert the class the unigrams imply, in the way
"not recommended" inverts "recommended". A unigram model cannot represent this at all. A
bigram model can, which is the point: the cheap competitor for order dependent signal is
not a transformer, it is one more line in a vectoriser.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BACKGROUND_SHARE = 0.55


@dataclass(frozen=True)
class Corpus:
    """Documents as token id sequences, with labels and the vocabulary that made them."""

    tokens: list[list[int]]
    labels: np.ndarray
    vocab_size: int
    n_classes: int
    phrase_rate: float

    def __len__(self) -> int:
        return len(self.tokens)

    def as_text(self) -> list[str]:
        """Token ids rendered as words, for the vectoriser based models.

        Ids are rendered as 'w123' rather than mapped to English. Inventing readable words
        would suggest the task carries meaning it does not have, and every model here sees
        exactly the same token stream either way.
        """
        return [" ".join(f"w{t}" for t in doc) for doc in self.tokens]


def _class_word_distributions(
    vocab_size: int, n_classes: int, rng: np.random.Generator
) -> np.ndarray:
    """One multinomial over the vocabulary per class, sharing a common background."""
    background = rng.dirichlet(np.full(vocab_size, 0.4))
    distributions = []
    for _ in range(n_classes):
        specific = rng.dirichlet(np.full(vocab_size, 0.1))
        mixed = BACKGROUND_SHARE * background + (1 - BACKGROUND_SHARE) * specific
        distributions.append(mixed / mixed.sum())
    return np.array(distributions)


def generate(
    n_docs: int = 12_000,
    vocab_size: int = 1_500,
    n_classes: int = 4,
    doc_length: int = 48,
    phrase_rate: float = 0.35,
    label_noise: float = 0.03,
    seed: int = 20260219,
) -> Corpus:
    """Build a corpus.

    phrase_rate is the share of documents carrying an order dependent marker that
    overrides the unigram signal. At 0 the task is pure bag of words. At 1 every label is
    decided by a word pair no unigram model can see.
    """
    rng = np.random.default_rng(seed)
    distributions = _class_word_distributions(vocab_size, n_classes, rng)

    # Marker pairs are drawn from the top of the vocabulary so they cannot collide with
    # the class specific words, which are spread across all of it.
    marker_first = vocab_size - 1
    marker_second = [vocab_size - 2 - c for c in range(n_classes)]

    tokens, labels = [], []
    for _ in range(n_docs):
        base_class = int(rng.integers(0, n_classes))
        document = list(rng.choice(vocab_size, size=doc_length, p=distributions[base_class]))
        label = base_class

        if rng.random() < phrase_rate:
            # The marker is the ordered pair (marker_first, marker_second[target]).
            # Scattering the same two ids in the wrong order carries no information, which
            # is what makes this invisible to a unigram model and visible to a bigram one.
            target = int(rng.integers(0, n_classes))
            position = int(rng.integers(0, doc_length - 1))
            document[position] = marker_first
            document[position + 1] = marker_second[target]
            label = target

        if rng.random() < label_noise:
            label = int(rng.integers(0, n_classes))

        tokens.append([int(t) for t in document])
        labels.append(label)

    return Corpus(tokens=tokens, labels=np.array(labels, dtype=np.int64),
                  vocab_size=vocab_size, n_classes=n_classes, phrase_rate=phrase_rate)


def split(corpus: Corpus, train_share: float = 0.7, seed: int = 20260219):
    """Split into train and evaluation halves, without shuffling the corpus in place."""
    rng = np.random.default_rng(seed + 99)
    order = rng.permutation(len(corpus))
    cut = int(len(corpus) * train_share)
    train_idx, eval_idx = order[:cut], order[cut:]

    def take(idx):
        return Corpus(
            tokens=[corpus.tokens[i] for i in idx],
            labels=corpus.labels[idx],
            vocab_size=corpus.vocab_size,
            n_classes=corpus.n_classes,
            phrase_rate=corpus.phrase_rate,
        )

    return take(train_idx), take(eval_idx)
