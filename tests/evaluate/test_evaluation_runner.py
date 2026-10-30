"""Generic lifecycle engine: ordering, reservation boundary, guard and outcome.

The runner is exercised with fake tasks and predictors (it has no task logic to
test), and with the output reservation replaced by a recorder so no files are
written. The production frozen-probe path is covered by the CLI integration test.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from modules.evaluate.src.evaluation import runner
from modules.evaluate.src.evaluation.contracts import (
    FitOutcome,
    FitSummary,
    PredictionBundle,
    ScoringResult,
)
from modules.evaluate.src.evaluation.runner import (
    SingleProcessRequired,
    run_evaluation,
)


class _Accelerator:
    def __init__(self, *, main=True, num_processes=1):
        self.is_main_process = main
        self.num_processes = num_processes

    def wait_for_everyone(self):
        pass


class FakePredictor:
    name = "fake_predictor"
    transfers_to_external = True

    def __init__(self, log, branches=("main",)):
        self.log = log
        self.branches = branches

    def fit(self, context, prepared):
        self.log.append(("fit",))
        return FitOutcome(
            summary=FitSummary(predictor=self.name, details={"n": prepared["n"]}),
            fitted={"n": prepared["n"], "fitted": True},
        )

    def predict(self, context, prepared):
        self.log.append(("predict", context.split, prepared.get("view")))
        return tuple(
            PredictionBundle(
                predictor=self.name,
                split=context.split,
                branch=branch,
                workspace=context.workspace,
                payload=prepared,
            )
            for branch in self.branches
        )


class FakeTask:
    name = "fake"

    def __init__(self, log, *, single=False, fail_in=None, branches=("main",)):
        self.log = log
        self.requires_single_process = single
        self.fail_in = fail_in
        self.branches = branches

    def prepare(self, context):
        self.log.append(("prepare",))
        if self.fail_in == "prepare":
            raise RuntimeError("prepare failed")
        return {"n": 3}

    def prepare_external(self, context, fitted, external):
        self.log.append(("prepare_external", external.name))
        assert fitted["fitted"] is True, "external view must reuse the fitted state"
        return {"n": fitted["n"], "view": external.name}

    def predictor(self, context):
        return FakePredictor(self.log, branches=self.branches)

    def score(self, context, bundle):
        self.log.append(("score", bundle.split, bundle.branch))
        return ScoringResult(
            predictor=bundle.predictor,
            split=bundle.split,
            branch=bundle.branch,
            payload={"ok": True},
        )

    def finalize_view(self, context, predictions, scores):
        self.log.append(("finalize", context.split, tuple(s.branch for s in scores)))


def _config(*external_names):
    return SimpleNamespace(
        external_test_sets=[
            SimpleNamespace(name=name, dataset_repo_id=f"org/{name}")
            for name in external_names
        ]
    )


@pytest.fixture
def recorded_reservation(monkeypatch):
    """Replace output reservation by a recorder; returns the event log."""
    log: list = []

    @contextmanager
    def reserve(config, workspace, accelerator, *, overwrite):
        log.append(("reserve-enter",))
        try:
            yield
        finally:
            log.append(("reserve-exit",))

    monkeypatch.setattr(runner, "reserve_outputs", reserve)
    monkeypatch.setattr(runner, "get_external_workspace", _fake_external_workspace)
    return log


class _FakeWorkspace:
    def setup_directories(self):
        pass


def _fake_external_workspace(workspace, external):
    return _FakeWorkspace()


def test_all_work_runs_inside_reservation_in_order(recorded_reservation):
    log = recorded_reservation
    run_evaluation(
        _config("ext"), SimpleNamespace(), _Accelerator(), task=FakeTask(log)
    )
    assert log == [
        ("reserve-enter",),
        ("prepare",),
        ("fit",),
        ("predict", "test", None),
        ("score", "test", "main"),
        ("finalize", "test", ("main",)),
        ("prepare_external", "ext"),
        ("predict", "ext", "ext"),
        ("score", "ext", "main"),
        ("finalize", "ext", ("main",)),
        ("reserve-exit",),
    ]


def test_external_views_are_keyed_by_name_and_fitted_once(recorded_reservation):
    log = recorded_reservation
    outcome = run_evaluation(
        _config("a", "b"), SimpleNamespace(), _Accelerator(), task=FakeTask(log)
    )
    assert outcome.fit.predictor == "fake_predictor"
    assert outcome.fit.details == {"n": 3}
    assert set(outcome.external) == {"a", "b"}
    assert outcome.external["a"].predictions[0].split == "a"
    assert log.count(("fit",)) == 1


def test_predictors_without_fitted_state_skip_external_views(recorded_reservation):
    log = recorded_reservation

    class NonTransferringPredictor(FakePredictor):
        transfers_to_external = False

    class Task(FakeTask):
        def predictor(self, context):
            return NonTransferringPredictor(self.log)

    outcome = run_evaluation(
        _config("a", "b"), SimpleNamespace(), _Accelerator(), task=Task(log)
    )
    assert outcome.external == {}
    assert ("prepare_external",) not in log
    assert log.count(("score", "test", "main")) == 1


def test_scores_only_on_main_process(recorded_reservation):
    log = recorded_reservation
    outcome = run_evaluation(
        _config(), SimpleNamespace(), _Accelerator(main=False), task=FakeTask(log)
    )
    assert outcome.main.scores is None
    assert ("predict", "test", None) in log
    assert ("score", "test", "main") not in log


def test_single_process_guard_fails_before_reservation(recorded_reservation):
    log = recorded_reservation
    with pytest.raises(SingleProcessRequired, match="single-process"):
        run_evaluation(
            _config(),
            SimpleNamespace(),
            _Accelerator(num_processes=2),
            task=FakeTask(log, single=True),
        )
    assert log == []


def test_reservation_rejection_stops_before_model_work(monkeypatch):
    log: list = []

    class Rejected(Exception):
        pass

    @contextmanager
    def reject(config, workspace, accelerator, *, overwrite):
        log.append("reserve-rejected")
        raise Rejected("output exists")
        yield  # pragma: no cover

    monkeypatch.setattr(runner, "reserve_outputs", reject)
    with pytest.raises(Rejected):
        run_evaluation(_config(), SimpleNamespace(), _Accelerator(), task=FakeTask(log))
    assert log == ["reserve-rejected"]


def test_prepare_failure_exits_reservation_and_propagates(recorded_reservation):
    log = recorded_reservation
    with pytest.raises(RuntimeError, match="prepare failed"):
        run_evaluation(
            _config(),
            SimpleNamespace(),
            _Accelerator(),
            task=FakeTask(log, fail_in="prepare"),
        )
    assert log == [("reserve-enter",), ("prepare",), ("reserve-exit",)]


def test_every_predicted_branch_is_scored_in_order(recorded_reservation):
    log = recorded_reservation
    outcome = run_evaluation(
        _config(),
        SimpleNamespace(),
        _Accelerator(),
        task=FakeTask(log, branches=("layer_0", "layer_1", "random_trunk")),
    )
    assert [bundle.branch for bundle in outcome.main.predictions] == [
        "layer_0",
        "layer_1",
        "random_trunk",
    ]
    assert [result.branch for result in outcome.main.scores] == [
        "layer_0",
        "layer_1",
        "random_trunk",
    ]
    assert [entry for entry in log if entry[0] == "score"] == [
        ("score", "test", "layer_0"),
        ("score", "test", "layer_1"),
        ("score", "test", "random_trunk"),
    ]


def test_finalize_runs_once_per_view_after_scoring_on_main(recorded_reservation):
    log = recorded_reservation
    run_evaluation(
        _config(),
        SimpleNamespace(),
        _Accelerator(),
        task=FakeTask(log, branches=("layer_0", "layer_1")),
    )
    finalize = [entry for entry in log if entry[0] == "finalize"]
    assert finalize == [("finalize", "test", ("layer_0", "layer_1"))]
    assert log.index(("score", "test", "layer_1")) < log.index(finalize[0])


def test_finalize_is_skipped_on_non_main_process(recorded_reservation):
    log = recorded_reservation
    outcome = run_evaluation(
        _config(),
        SimpleNamespace(),
        _Accelerator(main=False),
        task=FakeTask(log, branches=("layer_0", "layer_1")),
    )
    assert not [entry for entry in log if entry[0] in {"score", "finalize"}]
    assert outcome.main.scores is None
