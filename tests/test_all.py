"""Tests for inference-economics-bench.

The weight is on the parts that decide a published claim: the corpus really contains the
signal it claims to, the economics really implement dominance, the pricing path really
falls back, and the latency harness really discards warmups.

The transformer is trained in exactly two tests, both marked slow. Training it costs most
of a minute and the properties worth asserting about it do not need sixteen repetitions.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from botocore.stub import Stubber

from inference_economics import economics, latency, pricing
from inference_economics.config import BenchConfig, ConfigError
from inference_economics.corpus import generate, split
from inference_economics.models import MODEL_NAMES, build


def point(model="m", accuracy=0.9, per_item_ms=1.0, p99_ms=2.0,
          usd_per_million=0.025, size_bytes=1000):
    return economics.Point(model, accuracy, per_item_ms, p99_ms, usd_per_million, size_bytes)


def _report_with(rows):
    """Build a genuine Report from fabricated measurements.

    A real Report rather than a stand in, because Report is a frozen dataclass and
    subclassing it to override one field fails in a way that teaches nothing. Costs are
    left to the real pricing path, so the rendered table is the one users see.
    """
    from inference_economics.report import ModelResult, Report

    results = []
    for name, accuracy, p99_ms, per_item_ms in rows:
        profile = latency.LatencyProfile(
            model=name, batch_size=1, samples=100, p50_ms=per_item_ms,
            p95_ms=p99_ms * 0.9, p99_ms=p99_ms, mean_ms=per_item_ms,
            per_item_ms=per_item_ms,
        )
        results.append(ModelResult(name, accuracy, 0.0, 1000, {1: profile}))
    return Report(config=BenchConfig(), price=pricing.from_snapshot(), results=results)


class TestCorpusContainsTheSignalItClaims:
    """The whole comparison rests on this. If the task has no order dependent signal, the
    expensive models are dominated by construction and the benchmark is a strawman."""

    def test_it_is_deterministic(self):
        a, b = generate(n_docs=500, seed=7), generate(n_docs=500, seed=7)
        assert a.tokens == b.tokens
        assert np.array_equal(a.labels, b.labels)

    def test_a_different_seed_gives_a_different_corpus(self):
        assert generate(n_docs=500, seed=7).tokens != generate(n_docs=500, seed=8).tokens

    def test_classes_are_roughly_balanced(self):
        counts = np.bincount(generate(n_docs=4000).labels)
        assert counts.min() / counts.max() > 0.8

    def test_at_zero_phrase_rate_bigrams_add_nothing(self):
        """The control. With no order dependent signal, a bigram model has nothing to find
        and any accuracy gain would mean the generator is leaking something else."""
        unigram, bigram = _accuracy_pair(phrase_rate=0.0)
        assert abs(bigram - unigram) < 0.01

    def test_with_phrase_signal_bigrams_beat_unigrams(self):
        unigram, bigram = _accuracy_pair(phrase_rate=0.7)
        assert bigram > unigram + 0.015

    def test_label_noise_caps_achievable_accuracy(self):
        corpus = generate(n_docs=3000, label_noise=0.4, phrase_rate=0.0)
        train, evaluation = split(corpus)
        model = build("logreg_unigram")
        model.fit(train)
        accuracy = np.mean(model.predict_batch(evaluation.as_text()) == evaluation.labels)
        assert accuracy < 0.85, "heavy label noise should make the task hard"

    def test_split_does_not_overlap(self):
        corpus = generate(n_docs=1000)
        train, evaluation = split(corpus, train_share=0.7)
        assert len(train) + len(evaluation) == len(corpus)
        assert len(train) == 700

    def test_rendered_text_round_trips_to_token_ids(self):
        from inference_economics.models import _parse

        corpus = generate(n_docs=20, doc_length=12)
        assert _parse(corpus.as_text()[0]) == corpus.tokens[0]


def _accuracy_pair(phrase_rate: float) -> tuple[float, float]:
    corpus = generate(n_docs=6000, phrase_rate=phrase_rate)
    train, evaluation = split(corpus)
    text = evaluation.as_text()
    out = []
    for name in ("logreg_unigram", "logreg_ngram2"):
        model = build(name)
        model.fit(train)
        out.append(float(np.mean(model.predict_batch(text) == evaluation.labels)))
    return out[0], out[1]


class TestModelsShareOneInterface:
    @pytest.mark.parametrize("name", [n for n in MODEL_NAMES if n != "transformer"])
    def test_every_model_fits_predicts_and_reports_a_size(self, name):
        corpus = generate(n_docs=1500)
        train, evaluation = split(corpus)
        model = build(name)
        model.fit(train)
        text = evaluation.as_text()

        batch = model.predict_batch(text)
        assert len(batch) == len(text)
        assert model.predict_one(text[0]) == batch[0]
        assert model.size_bytes() > 0

    def test_an_unknown_model_is_refused_by_name(self):
        with pytest.raises(ValueError, match="unknown model"):
            build("gpt5")

    def test_the_majority_baseline_predicts_the_training_mode(self):
        corpus = generate(n_docs=2000)
        train, _ = split(corpus)
        model = build("majority")
        model.fit(train)
        assert model.predict_one("w1 w2") == int(np.bincount(train.labels).argmax())

    @pytest.mark.slow
    def test_the_transformer_learns_something(self):
        corpus = generate(n_docs=4000)
        train, evaluation = split(corpus)
        model = build("transformer")
        model.fit(train)
        accuracy = np.mean(model.predict_batch(evaluation.as_text()) == evaluation.labels)
        assert accuracy > 0.5, "should comfortably beat a four class baseline"

    @pytest.mark.slow
    def test_the_transformer_is_charged_for_parsing_like_everyone_else(self):
        """Both prediction paths take a string, so no model is handed a head start."""
        corpus = generate(n_docs=1500)
        train, evaluation = split(corpus)
        model = build("transformer")
        model.fit(train)
        text = evaluation.as_text()
        assert model.predict_one(text[0]) == model.predict_batch(text[:1])[0]


class TestDominanceIsWhatItClaims:
    def test_cheaper_and_more_accurate_dominates(self):
        better = point("a", accuracy=0.9, usd_per_million=0.01)
        worse = point("b", accuracy=0.8, usd_per_million=0.02)
        assert economics.dominates(better, worse)
        assert not economics.dominates(worse, better)

    def test_equal_on_both_axes_dominates_neither_way(self):
        a, b = point("a"), point("b")
        assert not economics.dominates(a, b) and not economics.dominates(b, a)

    def test_more_accurate_at_the_same_price_dominates(self):
        assert economics.dominates(point("a", accuracy=0.9, usd_per_million=0.01),
                                   point("b", accuracy=0.8, usd_per_million=0.01))

    def test_more_accurate_and_dearer_dominates_nothing(self):
        """The interesting case: a genuine trade off, where budget decides."""
        a = point("a", accuracy=0.9, usd_per_million=0.10)
        b = point("b", accuracy=0.8, usd_per_million=0.01)
        assert not economics.dominates(a, b) and not economics.dominates(b, a)

    def test_the_frontier_excludes_dominated_points(self):
        points = [point("cheap", accuracy=0.8, usd_per_million=0.01),
                  point("mid", accuracy=0.9, usd_per_million=0.05),
                  point("silly", accuracy=0.7, usd_per_million=0.20)]
        assert [p.model for p in economics.frontier(points)] == ["cheap", "mid"]

    def test_dominated_reports_who_does_the_beating(self):
        points = [point("cheap", accuracy=0.8, usd_per_million=0.01),
                  point("silly", accuracy=0.7, usd_per_million=0.20)]
        beaten, by = economics.dominated(points)[0]
        assert beaten.model == "silly" and by.model == "cheap"

    def test_the_frontier_is_ordered_by_cost(self):
        points = [point("b", accuracy=0.9, usd_per_million=0.05),
                  point("a", accuracy=0.8, usd_per_million=0.01)]
        assert [p.usd_per_million for p in economics.frontier(points)] == [0.01, 0.05]


class TestBudgetAndSlaAreDifferentQuestions:
    def _points(self):
        return [point("cheap", accuracy=0.80, p99_ms=0.5, usd_per_million=0.01),
                point("mid", accuracy=0.90, p99_ms=9.0, usd_per_million=0.05),
                point("dear", accuracy=0.95, p99_ms=2.0, usd_per_million=0.50)]

    def test_budget_picks_the_best_affordable_model(self):
        assert economics.best_within_budget(self._points(), 0.10).model == "mid"

    def test_a_budget_below_everything_returns_nothing(self):
        assert economics.best_within_budget(self._points(), 0.001) is None

    def test_an_sla_can_exclude_the_model_the_budget_would_pick(self):
        """The reason both exist. A tail latency limit rules out the mid model, which is
        exactly what a budget of 0.10 would have chosen."""
        chosen = economics.best_within_sla(self._points(), 3.0)
        assert chosen.model == "dear"
        assert economics.best_within_budget(self._points(), 0.10).model == "mid"

    def test_an_impossible_sla_returns_nothing(self):
        assert economics.best_within_sla(self._points(), 0.01) is None

    def test_accuracy_per_extra_dollar(self):
        cheap = point("a", accuracy=0.80, usd_per_million=0.01)
        dear = point("b", accuracy=0.90, usd_per_million=0.11)
        assert economics.accuracy_per_extra_dollar(cheap, dear) == pytest.approx(100.0)

    def test_a_free_upgrade_is_infinite_value_not_an_error(self):
        cheap = point("a", accuracy=0.80, usd_per_million=0.05)
        dear = point("b", accuracy=0.90, usd_per_million=0.05)
        assert economics.accuracy_per_extra_dollar(cheap, dear) == float("inf")

    def test_cost_conversion(self):
        """One millisecond per item on a $0.08925 per hour box."""
        assert economics.usd_per_million(1.0, 0.08925) == pytest.approx(0.0247917, rel=1e-4)


class TestPricing:
    def test_the_snapshot_carries_its_date(self):
        price = pricing.from_snapshot("c7i.large")
        assert price.source == "snapshot" and price.as_of == pricing.SNAPSHOT_DATE

    def test_an_unpriced_instance_is_refused_by_name(self):
        with pytest.raises(pricing.PricingError, match="no snapshot price"):
            pricing.from_snapshot("x1e.32xlarge")

    def test_a_stubbed_aws_response_is_parsed(self):
        import boto3

        client = boto3.client("pricing", region_name="us-east-1",
                              aws_access_key_id="x", aws_secret_access_key="y")
        document = json.dumps({"terms": {"OnDemand": {"offer": {"priceDimensions": {
            "rate": {"pricePerUnit": {"USD": "0.1234000000"}}}}}}})
        with Stubber(client) as stub:
            stub.add_response("get_products", {"PriceList": [document]})
            price = pricing.from_aws("c7i.large", client=client)
        assert price.usd_per_hour == pytest.approx(0.1234)
        assert price.source == "aws-price-list"

    def test_an_empty_aws_response_falls_back_to_the_snapshot(self):
        import boto3

        client = boto3.client("pricing", region_name="us-east-1",
                              aws_access_key_id="x", aws_secret_access_key="y")
        with Stubber(client) as stub:
            stub.add_response("get_products", {"PriceList": []})
            price = pricing.from_aws("c7i.large", client=client)
        assert price.source == "snapshot"

    def test_fallback_can_be_refused(self):
        """A caller who needs a live price must be able to insist on one."""
        import boto3

        client = boto3.client("pricing", region_name="us-east-1",
                              aws_access_key_id="x", aws_secret_access_key="y")
        with Stubber(client) as stub:
            stub.add_response("get_products", {"PriceList": []})
            with pytest.raises(pricing.PricingError, match="could not price"):
                pricing.from_aws("c7i.large", client=client, fall_back=False)

    def test_a_price_list_record_with_no_usd_is_refused(self):
        document = json.dumps({"terms": {"OnDemand": {"o": {"priceDimensions": {
            "r": {"pricePerUnit": {"USD": "0.0000000000"}}}}}}})
        with pytest.raises(pricing.PricingError, match="no non zero"):
            pricing._extract_on_demand_usd(document)

    def test_every_snapshot_price_is_positive(self):
        assert all(v > 0 for v in pricing.PRICE_SNAPSHOT_USD_PER_HOUR.values())


class TestLatencyHarness:
    class _Fake:
        name = "fake"

        def __init__(self):
            self.calls = 0

        def predict_one(self, document):
            self.calls += 1
            return 0

        def predict_batch(self, documents):
            self.calls += 1
            return np.zeros(len(documents), dtype=np.int64)

    def test_warmups_are_run_and_excluded_from_the_samples(self):
        model = self._Fake()
        profile = latency.measure(model, ["a b", "c d", "e f"], repeats=60, warmups=5)
        assert profile.samples == 60
        assert model.calls == 65, "warmups should run but not be timed"

    def test_percentiles_are_ordered(self):
        profile = latency.measure(self._Fake(), ["a b"] * 5, repeats=60, warmups=2)
        assert profile.p50_ms <= profile.p95_ms <= profile.p99_ms

    def test_per_item_cost_divides_by_batch_size(self):
        profile = latency.measure(self._Fake(), ["a b"] * 100, batch_size=10,
                                  repeats=60, warmups=2)
        assert profile.per_item_ms == pytest.approx(profile.p50_ms / 10)

    def test_items_per_second_is_the_reciprocal(self):
        profile = latency.measure(self._Fake(), ["a b"] * 20, repeats=60, warmups=2)
        assert profile.items_per_second == pytest.approx(1000.0 / profile.per_item_ms)

    def test_a_profile_is_json_serialisable(self):
        json.dumps(latency.measure(self._Fake(), ["a b"], repeats=60, warmups=2).as_dict())


class TestConfigRefusesUntrustworthyRuns:
    def test_a_corpus_too_small_to_split(self):
        with pytest.raises(ConfigError, match="too small"):
            BenchConfig(n_docs=100)

    def test_phrase_rate_outside_zero_to_one(self):
        with pytest.raises(ConfigError, match="between 0 and 1"):
            BenchConfig(phrase_rate=1.5)

    def test_label_noise_at_or_above_a_half(self):
        with pytest.raises(ConfigError, match="less information than a coin"):
            BenchConfig(label_noise=0.5)

    def test_an_extreme_train_share(self):
        with pytest.raises(ConfigError, match="between 0.1 and 0.9"):
            BenchConfig(train_share=0.95)

    def test_an_unknown_model(self):
        with pytest.raises(ConfigError, match="unknown models"):
            BenchConfig(models=("bert",))

    def test_no_models_at_all(self):
        with pytest.raises(ConfigError, match="at least one model"):
            BenchConfig(models=())

    def test_a_zero_batch_size(self):
        with pytest.raises(ConfigError, match="at least 1"):
            BenchConfig(batch_sizes=(0,))

    def test_too_few_latency_samples_for_a_p99(self):
        with pytest.raises(ConfigError, match="below 50"):
            BenchConfig(latency_repeats=10)

    def test_no_warmups(self):
        with pytest.raises(ConfigError, match="at least one warmup"):
            BenchConfig(latency_warmups=0)

    def test_an_unpriced_instance(self):
        with pytest.raises(ConfigError, match="no snapshot price"):
            BenchConfig(instance_type="x1e.32xlarge")

    def test_as_dict_round_trips(self):
        assert BenchConfig().as_dict()["instance_type"] == "c7i.large"


class TestExitCodes:
    """The verdicts, tested against constructed reports rather than real training.

    Building the points directly is deliberate: these assertions are about the decision
    rule, and training five models to produce one would make the test slow and its failure
    ambiguous between the rule and the models.
    """

    def _report(self, points):
        from inference_economics.cli import Report as _  # noqa: F401
        from inference_economics.report import Report

        class Fake(Report):
            def __init__(self, pts):
                self._pts = pts

            def points(self, batch_size=1):
                return self._pts

        return Fake(points)

    def _three(self):
        return [point("cheap", accuracy=0.80, p99_ms=0.5, usd_per_million=0.01),
                point("mid", accuracy=0.90, p99_ms=2.0, usd_per_million=0.05),
                point("silly", accuracy=0.70, p99_ms=9.0, usd_per_million=0.50)]

    def test_a_frontier_model_exits_zero(self):
        from inference_economics import cli

        assert cli.classify(self._report(self._three()), "mid", 1, None) == cli.EXIT_ON_FRONTIER

    def test_a_dominated_model_exits_four(self):
        from inference_economics import cli

        assert cli.classify(self._report(self._three()), "silly", 1,
                            None) == cli.EXIT_DOMINATED

    def test_paying_more_for_a_negligible_gain_exits_one(self):
        from inference_economics import cli

        points = [point("cheap", accuracy=0.900, p99_ms=0.5, usd_per_million=0.01),
                  point("dear", accuracy=0.905, p99_ms=1.0, usd_per_million=0.50)]
        assert cli.classify(self._report(points), "dear", 1, None) == cli.EXIT_OVERPAYING

    def test_a_real_accuracy_gain_is_not_overpaying(self):
        from inference_economics import cli

        points = [point("cheap", accuracy=0.80, p99_ms=0.5, usd_per_million=0.01),
                  point("dear", accuracy=0.95, p99_ms=1.0, usd_per_million=0.50)]
        assert cli.classify(self._report(points), "dear", 1, None) == cli.EXIT_ON_FRONTIER

    def test_an_sla_nothing_meets_exits_five(self):
        from inference_economics import cli

        assert cli.classify(self._report(self._three()), "mid", 1,
                            0.01) == cli.EXIT_NOTHING_MEETS_SLA

    def test_a_choice_that_misses_a_satisfiable_sla_exits_one(self):
        from inference_economics import cli

        assert cli.classify(self._report(self._three()), "silly", 1,
                            1.0) == cli.EXIT_OVERPAYING

    def test_the_sla_is_checked_before_dominance(self):
        """A model that breaches the SLA is reported as that, not as dominated. The SLA is
        the hard constraint and naming the wrong reason sends somebody down the wrong path."""
        from inference_economics import cli

        code = cli.classify(self._report(self._three()), "silly", 1, 1.0)
        assert code == cli.EXIT_OVERPAYING and code != cli.EXIT_DOMINATED


class TestCliEndToEnd:
    def test_a_rejected_config_exits_two(self, capsys):
        from inference_economics import cli

        assert cli.main(["--n-docs", "50"]) == cli.EXIT_BAD_CONFIG
        assert "invalid benchmark" in capsys.readouterr().err

    def test_a_small_real_run_prints_a_table(self, capsys):
        from inference_economics import cli

        code = cli.main(["--n-docs", "1200", "--latency-repeats", "50",
                         "--choose", "logreg_unigram"])
        out = capsys.readouterr().out
        assert "on the frontier" in out and "logreg_unigram" in out
        assert code in {cli.EXIT_ON_FRONTIER, cli.EXIT_OVERPAYING, cli.EXIT_DOMINATED}

    def test_json_output_parses(self, capsys):
        from inference_economics import cli

        cli.main(["--n-docs", "1200", "--latency-repeats", "50", "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert "frontier_batch1" in payload and payload["price"]["source"] == "snapshot"

    def test_an_internal_failure_exits_three(self, monkeypatch, capsys):
        from inference_economics import cli

        def boom(*args, **kwargs):
            raise RuntimeError("training exploded")

        monkeypatch.setattr(cli, "run", boom)
        code = cli.main(["--n-docs", "1200", "--latency-repeats", "50"])
        assert code == cli.EXIT_INTERNAL_ERROR
        assert "failed to complete" in capsys.readouterr().err

    def test_the_rendered_table_names_the_price_source(self, capsys):
        from inference_economics import cli

        cli.main(["--n-docs", "1200", "--latency-repeats", "50"])
        assert "snapshot" in capsys.readouterr().out


class TestReportWiring:
    def test_a_report_prices_every_model_it_ran(self):
        from inference_economics.report import run

        config = BenchConfig(n_docs=1200, models=("majority", "logreg_unigram"),
                             batch_sizes=(1, 4), latency_repeats=50)
        report = run(config)
        assert len(report.results) == 2
        assert {p.model for p in report.points(1)} == {"majority", "logreg_unigram"}
        assert set(report.results[0].profiles) == {1, 4}

    def test_the_report_dict_is_json_serialisable(self):
        from inference_economics.report import run

        config = BenchConfig(n_docs=1200, models=("majority",), batch_sizes=(1,),
                             latency_repeats=50)
        json.dumps(run(config).as_dict())

    def test_a_supplied_price_is_used_rather_than_the_snapshot(self):
        from inference_economics.report import run

        config = BenchConfig(n_docs=1200, models=("majority",), batch_sizes=(1,),
                             latency_repeats=50)
        custom = pricing.Price("c7i.large", 1.0, source="test", as_of="now")
        report = run(config, custom)
        assert report.price.source == "test"
        assert report.points(1)[0].usd_per_million == pytest.approx(
            economics.usd_per_million(report.results[0].profiles[1].per_item_ms, 1.0)
        )


class TestRenderingTheSlaVerdict:
    def test_the_sla_lines_are_rendered_both_ways(self, capsys):
        """Both branches print, because a reader needs to know which case they are in."""
        from inference_economics import cli

        report = _report_with(
            [("cheap", 0.80, 0.5, 0.4032), ("dear", 0.95, 4.0, 20.16)]
        )

        assert [p.model for p in report.points(1)] == ["cheap", "dear"]
        satisfiable = cli.render(report, "cheap", 1, 1.0)
        assert "most accurate model inside a 1.00 ms p99 budget: cheap" in satisfiable

        impossible = cli.render(report, "cheap", 1, 0.01)
        assert "no model meets a p99 budget of 0.01 ms" in impossible

    def test_a_point_serialises(self):
        payload = point("m").as_dict()
        assert payload["model"] == "m" and "usd_per_million" in payload
        json.dumps(payload)

    def test_a_client_that_cannot_be_constructed_falls_back(self, monkeypatch):
        """The fallback exists so an unreachable pricing endpoint cannot fail a benchmark.

        The failure is injected rather than produced by really having no network. An
        earlier version of this test called from_aws with no client and let it fail for
        real, which passed everywhere and made coverage depend on the environment: if
        boto3.client itself raised, the get_products call was never reached and those
        lines went uncovered, so the same suite reported 100 percent on a machine with
        network and 99 without. A clean clone verified offline is what surfaced it.
        """
        import boto3

        def no_client(*args, **kwargs):
            raise RuntimeError("endpoint unreachable")

        monkeypatch.setattr(boto3, "client", no_client)
        price = pricing.from_aws("c7i.large")
        assert price.source == "snapshot"

    def test_a_client_that_cannot_be_constructed_can_also_refuse_to_fall_back(
        self, monkeypatch
    ):
        import boto3

        def no_client(*args, **kwargs):
            raise RuntimeError("endpoint unreachable")

        monkeypatch.setattr(boto3, "client", no_client)
        with pytest.raises(pricing.PricingError, match="could not price"):
            pricing.from_aws("c7i.large", fall_back=False)
