"""
Tests for the Phase-1 abstractions extracted from navsim_manager.py.

These exercise the *generic* shape of each abstraction — anything
project-specific (NAVSIM column maps, backbone tables, etc.) lives in
the manager subclass and is not under test here.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# script_utils
# ---------------------------------------------------------------------------


def test_quote_bash_escapes_single_quotes():
    from expflow.script_utils import quote_bash

    assert quote_bash("plain") == "plain"
    assert quote_bash("a b") == "'a b'"
    assert quote_bash("a'b; rm -rf /") == "'a'\"'\"'b; rm -rf /'"


def test_assert_safe_identifier_rejects_injection():
    from expflow.script_utils import assert_safe_identifier

    assert assert_safe_identifier("table_name") == "table_name"
    with pytest.raises(ValueError):
        assert_safe_identifier("users; DROP TABLE x")
    with pytest.raises(ValueError):
        assert_safe_identifier("0starts_with_digit")
    with pytest.raises(ValueError):
        assert_safe_identifier("")


def test_git_worktree_block_no_branch_is_noop():
    from expflow.script_utils import git_worktree_block

    out = git_worktree_block({"exp_id": "X"})
    assert "no git_branch" in out
    assert "git worktree add" not in out


def test_git_worktree_block_emits_trap():
    from expflow.script_utils import git_worktree_block

    out = git_worktree_block(
        {"exp_id": "X", "git_branch": "main"}, devkit_subdir="src"
    )
    assert "git worktree add" in out
    assert "trap cleanup_worktree EXIT" in out
    assert '"${WORKTREE_DIR}/src"' in out


def test_git_worktree_block_rejects_unsafe_subdir():
    from expflow.script_utils import git_worktree_block

    with pytest.raises(ValueError):
        git_worktree_block(
            {"exp_id": "X", "git_branch": "main"},
            devkit_subdir='"; rm -rf /; echo "',
        )


# ---------------------------------------------------------------------------
# result_record
# ---------------------------------------------------------------------------


def test_result_record_builder_sections():
    from expflow.result_record import ResultRecordBuilder

    record = (
        ResultRecordBuilder()
        .core_section("e1", "completed", created_at="2026-01-01T00:00:00")
        .git_section({"git_commit": "abc", "git_branch": "main", "git_dirty": False})
        .slurm_section(
            {"partition": "cpu", "num_gpus": 0}, {"train_job_id": "12345"}
        )
        .custom_section("training", {"epochs": 5})
        .build()
    )
    assert record["exp_id"] == "e1"
    assert record["status"] == "completed"
    assert record["git"]["commit"] == "abc"
    assert record["slurm"]["train_job_id"] == "12345"
    assert record["training"] == {"epochs": 5}
    assert record["stored_at"].endswith("Z")


def test_noop_enricher_returns_record_unchanged():
    from expflow.result_record import NoopRecordEnricher

    rec = {"a": 1}
    out = NoopRecordEnricher().enrich(rec, {}, {})
    assert out is rec


# ---------------------------------------------------------------------------
# scope_aggregator
# ---------------------------------------------------------------------------


def _write_eval_csv(directory: Path, score: float, scenarios: int):
    directory.mkdir(parents=True)
    with open(directory / "results.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["token", "score"])
        for i in range(scenarios):
            w.writerow([f"t{i}", score])


def test_scope_aggregator_overall_wins_when_present(tmp_path):
    from expflow.scope_aggregator import BaseScopeAggregator

    class Agg(BaseScopeAggregator):
        def known_scopes(self):
            return {"a", "b", "all"}

        def column_map(self):
            return {"score": "score"}

    _write_eval_csv(tmp_path / "exp_eval_test_a_20260101", 0.8, 10)
    _write_eval_csv(tmp_path / "exp_eval_test_b_20260101", 0.6, 10)
    _write_eval_csv(tmp_path / "exp_eval_test_all_20260101", 0.7, 20)

    res = Agg().aggregate("exp", tmp_path)
    assert res["primary_score"] == pytest.approx(0.7)
    # weighted of per-scope is also computed
    assert res["weighted_scores"]["score"] == pytest.approx(0.7)


def test_scope_aggregator_weighted_when_no_overall(tmp_path):
    from expflow.scope_aggregator import BaseScopeAggregator

    class Agg(BaseScopeAggregator):
        def known_scopes(self):
            return {"a", "b", "all"}

        def column_map(self):
            return {"score": "score"}

    _write_eval_csv(tmp_path / "exp_eval_test_a_20260101", 0.8, 100)
    _write_eval_csv(tmp_path / "exp_eval_test_b_20260101", 0.6, 50)

    res = Agg().aggregate("exp", tmp_path)
    # 0.8*100 + 0.6*50 = 110; / 150 = 0.7333
    assert res["primary_score"] == pytest.approx((0.8 * 100 + 0.6 * 50) / 150)
    assert res["overall_scores"] is None


def test_scope_aggregator_strict_id_match(tmp_path):
    """exp_id A3 should NOT match a sibling like A3-b."""
    from expflow.scope_aggregator import BaseScopeAggregator

    class Agg(BaseScopeAggregator):
        def known_scopes(self):
            return {"all"}

        def column_map(self):
            return {"score": "score"}

    _write_eval_csv(tmp_path / "A3_eval_test_all_20260101", 0.5, 10)
    _write_eval_csv(tmp_path / "A3-b_eval_test_all_20260101", 0.99, 10)
    res = Agg().aggregate("A3", tmp_path)
    # only A3 (not A3-b) contributes
    assert res["overall_scores"]["score"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# run_history (AttemptGrouping)
# ---------------------------------------------------------------------------


def test_attempt_grouping_id_is_deterministic():
    from expflow.run_history import AttemptGrouping

    g = AttemptGrouping(["checkpoint_path", "config.eval_type"])
    rec = {"checkpoint_path": "/x.ckpt", "config": {"eval_type": "one_stage"}}
    a = g.group_id("exp1", rec)
    b = g.group_id("exp1", dict(rec))
    assert a == b
    # different exp_id → different group
    assert g.group_id("exp2", rec) != a


def test_attempt_grouping_assigns_orders_and_summaries():
    from expflow.run_history import AttemptGrouping

    g = AttemptGrouping(["checkpoint_path"])
    runs = [
        {
            "exp_id": "X",
            "run_id": "r1",
            "attempt_group_id": g.group_id("X", {"checkpoint_path": "/x"}),
            "attempt_timestamp": "20260101_120000",
            "avg_pdms": 0.7,
        },
        {
            "exp_id": "X",
            "run_id": "r2",
            "attempt_group_id": g.group_id("X", {"checkpoint_path": "/x"}),
            "attempt_timestamp": "20260102_120000",
            "avg_pdms": 0.85,
        },
    ]
    g.assign_orders(runs)
    assert [r["attempt_order"] for r in runs] == [1, 2]

    summaries = g.build_summaries(runs)
    assert len(summaries) == 1
    s = summaries[0]
    assert s["attempts_count"] == 2
    assert s["best_attempt_id"] == "r2"
    assert s["mean_score"] == pytest.approx((0.7 + 0.85) / 2)


def test_attempt_grouping_eval_batch_id_sorts_jobs():
    from expflow.run_history import AttemptGrouping

    a = AttemptGrouping.eval_batch_id(["12", "11", "13"])
    b = AttemptGrouping.eval_batch_id(["13", "12", "11"])
    assert a == b == "batch__11_12_13"


# ---------------------------------------------------------------------------
# eval_log_parser
# ---------------------------------------------------------------------------


def test_eval_log_parser_extracts_pdms_and_status(tmp_path):
    from expflow.eval_log_parser import EvalLogParser

    log = tmp_path / "exp_eval_42.out"
    log.write_text(
        "loading checkpoint...\n"
        "Number of successful scenarios: 1234\n"
        "Final average score of valid results: 0.8472\n"
        "Status: SUCCESS\n"
    )
    p = EvalLogParser()
    out = p.parse(log)
    assert out["score"] == pytest.approx(0.8472)
    assert out["scenarios"] == 1234
    assert out["status"] == "SUCCESS"


def test_eval_log_parser_picks_latest_by_job_id(tmp_path):
    from expflow.eval_log_parser import EvalLogParser

    (tmp_path / "exp_eval_1.out").write_text("PDMS: 0.1\nStatus: SUCCESS\n")
    (tmp_path / "exp_eval_2.out").write_text("PDMS: 0.2\nStatus: SUCCESS\n")
    (tmp_path / "exp_eval_99.out").write_text("PDMS: 0.99\nStatus: SUCCESS\n")
    p = EvalLogParser()
    latest = p.latest_log(tmp_path, ["exp_eval_*.out"])
    assert latest.name == "exp_eval_99.out"


# ---------------------------------------------------------------------------
# eval_advisor
# ---------------------------------------------------------------------------


def test_default_eval_advisor_honours_overrides():
    from expflow.eval_advisor import DefaultEvalResourceAdvisor

    adv = DefaultEvalResourceAdvisor()
    out = adv.recommend({"eval_workers": 16, "eval_mem": "200G", "eval_time": "08:00:00"})
    assert out == {"workers": 16, "mem": "200G", "time": "08:00:00"}


def test_default_eval_advisor_caps_memory():
    from expflow.eval_advisor import DefaultEvalResourceAdvisor

    adv = DefaultEvalResourceAdvisor(mem_cap_gb=200, mem_per_worker_gb=4, mem_overhead_gb=32)
    # 64*4 + 32 = 288, capped at 200
    out = adv.recommend({"eval_workers": 64})
    assert out["mem"] == "200G"


# ---------------------------------------------------------------------------
# matrix_builder
# ---------------------------------------------------------------------------


class _StubMgr:
    def __init__(self):
        self.metadata = {}
        self.created = []
        self.submitted = []

    def create_experiment(self, exp_id, *, template=None, description="", **kw):
        self.created.append((exp_id, kw))
        self.metadata[exp_id] = {"status": "created"}

    def submit_experiment(self, exp_id, *, dry_run=False, **kw):
        self.submitted.append(exp_id)


def test_matrix_builder_enumerates_cartesian_product():
    from expflow.matrix_builder import MatrixExperimentBuilder

    mgr = _StubMgr()
    b = MatrixExperimentBuilder(
        mgr,
        axes={"lr": [1e-3, 1e-4], "seed": [1, 2, 3]},
        naming=lambda b: f"sweep_lr{b['lr']}_s{b['seed']}",
    )
    assert len(b.points()) == 6
    b.generate(dry_run=False)
    assert len(mgr.created) == 6
    assert mgr.created[0][1] == {"lr": 1e-3, "seed": 1}


def test_matrix_builder_skip_unregistered_on_submit():
    from expflow.matrix_builder import MatrixExperimentBuilder

    mgr = _StubMgr()
    b = MatrixExperimentBuilder(
        mgr,
        axes={"a": [1, 2]},
        naming=lambda x: f"e{x['a']}",
    )
    # only e1 registered — e2 is skipped
    mgr.metadata["e1"] = {"status": "created"}
    submitted = b.submit(dry_run=False)
    assert submitted == ["e1"]


# ---------------------------------------------------------------------------
# CheckpointResolver.resolve_for_eval
# ---------------------------------------------------------------------------


def test_resolve_for_eval_registry_first(tmp_path):
    from expflow.checkpoint_validator import CheckpointResolver

    reg = tmp_path / "registry"
    reg.mkdir()
    (reg / "exp1.txt").write_text("/some/path/epoch=42-step=100.ckpt")
    out = CheckpointResolver.resolve_for_eval("exp1", registry_dir=reg)
    assert out["source"] == "registry"
    assert out["checkpoint_epoch"] == 42


def test_resolve_for_eval_returns_none_when_no_source(tmp_path):
    from expflow.checkpoint_validator import CheckpointResolver

    out = CheckpointResolver.resolve_for_eval("nope", registry_dir=tmp_path)
    assert out["checkpoint_path"] is None
    assert out["source"] is None
