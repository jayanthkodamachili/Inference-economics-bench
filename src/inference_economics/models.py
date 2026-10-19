"""Five models behind one interface, spanning four orders of magnitude of inference cost.

The set is chosen so the comparison is a fair fight rather than a demonstration.

A majority class baseline is included because it is free, and because any model that does
not beat it is not a model. A unigram linear model is the cheap default. A bigram linear
model is the cheap competitor for order dependent signal, and it is here specifically to
stop the transformer winning by default: the honest alternative to a neural network on
word order is usually one more argument to the vectoriser, not a GPU.

Every model exposes the same two prediction paths, because they cost different amounts and
the difference decides deployments. predict_batch is throughput bound serving. predict_one
is a single request with nothing to amortise against, and it is where fixed per call costs
stop being invisible.
"""

from __future__ import annotations

import io
import pickle
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from inference_economics.corpus import Corpus

TRANSFORMER_SEED = 20260219


@dataclass
class Prediction:
    labels: np.ndarray


class Majority:
    """Predict the most common training label. The floor every other model must clear."""

    name = "majority"
    trainable = False

    def __init__(self) -> None:
        self._label = 0

    def fit(self, corpus: Corpus) -> None:
        self._label = int(np.bincount(corpus.labels).argmax())

    def predict_batch(self, documents: list[str]) -> np.ndarray:
        return np.full(len(documents), self._label, dtype=np.int64)

    def predict_one(self, document: str) -> int:
        return self._label

    def size_bytes(self) -> int:
        return len(pickle.dumps(self._label))


class LinearText:
    """TF-IDF into logistic regression, with the ngram range as the only difference."""

    def __init__(self, ngram_max: int = 1) -> None:
        self.name = "logreg_unigram" if ngram_max == 1 else f"logreg_ngram{ngram_max}"
        self.trainable = True
        self._ngram_max = ngram_max
        self._pipeline = None

    def fit(self, corpus: Corpus) -> None:
        self._pipeline = make_pipeline(
            TfidfVectorizer(ngram_range=(1, self._ngram_max), min_df=2),
            LogisticRegression(max_iter=1000),
        )
        self._pipeline.fit(corpus.as_text(), corpus.labels)

    def predict_batch(self, documents: list[str]) -> np.ndarray:
        return self._pipeline.predict(documents)

    def predict_one(self, document: str) -> int:
        return int(self._pipeline.predict([document])[0])

    def size_bytes(self) -> int:
        return len(pickle.dumps(self._pipeline))


class ReducedBoosting:
    """TF-IDF, reduced by SVD, into gradient boosted trees.

    The middle of the cost range, and a pipeline people really deploy. The SVD is not
    decoration: boosting on a sparse high dimensional TF-IDF matrix is slow enough that
    nobody does it, so leaving it out would have made this model a strawman too.
    """

    name = "svd_boosting"
    trainable = True

    def __init__(self, components: int = 128) -> None:
        self._components = components
        self._pipeline = None

    def fit(self, corpus: Corpus) -> None:
        self._pipeline = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2),
            TruncatedSVD(n_components=self._components, random_state=TRANSFORMER_SEED),
            HistGradientBoostingClassifier(max_iter=120, random_state=TRANSFORMER_SEED),
        )
        self._pipeline.fit(corpus.as_text(), corpus.labels)

    def predict_batch(self, documents: list[str]) -> np.ndarray:
        return self._pipeline.predict(documents)

    def predict_one(self, document: str) -> int:
        return int(self._pipeline.predict([document])[0])

    def size_bytes(self) -> int:
        return len(pickle.dumps(self._pipeline))


class TinyTransformer:
    """A small encoder trained from scratch, in PyTorch, on CPU.

    Trained from scratch rather than fine tuned from a checkpoint, because no pretrained
    weights could be downloaded in the environment this was built in. That is a real
    limitation and it is stated in the README rather than papered over: a fine tuned
    DistilBERT would be more accurate and considerably more expensive, which would move
    every number here without changing the argument.
    """

    name = "transformer"
    trainable = True

    def __init__(self, dim: int = 64, layers: int = 2, heads: int = 4,
                 max_length: int = 64, epochs: int = 6, batch_size: int = 64) -> None:
        self._dim, self._layers, self._heads = dim, layers, heads
        self._max_length, self._epochs, self._batch_size = max_length, epochs, batch_size
        self._model = None
        self._vocab_size = 0

    def _build(self, vocab_size: int, n_classes: int):
        import torch
        from torch import nn

        torch.manual_seed(TRANSFORMER_SEED)

        class Encoder(nn.Module):
            def __init__(self, dim, layers, heads, max_length, vocab, classes):
                super().__init__()
                self.embedding = nn.Embedding(vocab + 1, dim, padding_idx=vocab)
                self.position = nn.Parameter(torch.zeros(1, max_length, dim))
                layer = nn.TransformerEncoderLayer(
                    dim, heads, dim * 2, batch_first=True, dropout=0.1
                )
                # enable_nested_tensor is off because the nested tensor path is a
                # prototype that warns on every forward pass, and because it changes the
                # kernels used depending on the padding in a batch, which is exactly the
                # kind of input dependent variation a latency benchmark should not have.
                self.encoder = nn.TransformerEncoder(layer, layers,
                                                     enable_nested_tensor=False)
                self.head = nn.Linear(dim, classes)

            def forward(self, ids, mask):
                hidden = self.encoder(self.embedding(ids) + self.position,
                                      src_key_padding_mask=~mask)
                weights = mask.unsqueeze(-1).float()
                pooled = (hidden * weights).sum(1) / weights.sum(1).clamp(min=1.0)
                return self.head(pooled)

        return Encoder(self._dim, self._layers, self._heads, self._max_length,
                       vocab_size, n_classes)

    def _encode(self, token_lists: list[list[int]]):
        import torch

        pad = self._vocab_size
        ids = torch.full((len(token_lists), self._max_length), pad, dtype=torch.long)
        mask = torch.zeros((len(token_lists), self._max_length), dtype=torch.bool)
        for row, tokens in enumerate(token_lists):
            clipped = tokens[: self._max_length]
            ids[row, : len(clipped)] = torch.tensor(clipped, dtype=torch.long)
            mask[row, : len(clipped)] = True
        return ids, mask

    def fit(self, corpus: Corpus) -> None:
        import torch
        from torch import nn

        torch.manual_seed(TRANSFORMER_SEED)
        torch.set_num_threads(2)
        self._vocab_size = corpus.vocab_size
        self._model = self._build(corpus.vocab_size, corpus.n_classes)

        ids, mask = self._encode(corpus.tokens)
        labels = torch.tensor(corpus.labels, dtype=torch.long)
        optimiser = torch.optim.AdamW(self._model.parameters(), lr=3e-4)
        loss_fn = nn.CrossEntropyLoss()

        self._model.train()
        generator = torch.Generator().manual_seed(TRANSFORMER_SEED)
        for _ in range(self._epochs):
            order = torch.randperm(len(labels), generator=generator)
            for start in range(0, len(order), self._batch_size):
                batch = order[start : start + self._batch_size]
                optimiser.zero_grad()
                loss = loss_fn(self._model(ids[batch], mask[batch]), labels[batch])
                loss.backward()
                optimiser.step()
        self._model.eval()

    def predict_tokens(self, token_lists: list[list[int]]) -> np.ndarray:
        import torch

        ids, mask = self._encode(token_lists)
        with torch.no_grad():
            return self._model(ids, mask).argmax(dim=1).numpy()

    def predict_batch(self, documents: list[str]) -> np.ndarray:
        return self.predict_tokens([_parse(d) for d in documents])

    def predict_one(self, document: str) -> int:
        return int(self.predict_tokens([_parse(document)])[0])

    def size_bytes(self) -> int:
        import torch

        buffer = io.BytesIO()
        torch.save(self._model.state_dict(), buffer)
        return buffer.getbuffer().nbytes


def _parse(document: str) -> list[int]:
    """Turn the rendered text back into token ids.

    The transformer is charged for this. Every model is measured on the same input, a
    string, and whatever each one must do to consume it is part of its cost. Handing the
    neural model pre-tokenised tensors while the linear models parse text would be
    measuring two different tasks.
    """
    return [int(token[1:]) for token in document.split() if token.startswith("w")]


def build(name: str):
    """Construct a model by name."""
    builders = {
        "majority": Majority,
        "logreg_unigram": lambda: LinearText(ngram_max=1),
        "logreg_ngram2": lambda: LinearText(ngram_max=2),
        "svd_boosting": ReducedBoosting,
        "transformer": TinyTransformer,
    }
    if name not in builders:
        raise ValueError(f"unknown model {name!r}, expected one of {sorted(builders)}")
    return builders[name]()


MODEL_NAMES = ("majority", "logreg_unigram", "logreg_ngram2", "svd_boosting", "transformer")
