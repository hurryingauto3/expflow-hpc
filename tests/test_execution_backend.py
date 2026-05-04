"""
Tests for ExecutionBackend / LocalBackend / SlurmBackend (Phase 2).

LocalBackend gets full coverage since it actually runs on the test
machine. SlurmBackend tests just verify state-mapping and constructor
behaviour without invoking the SLURM CLI.
"""

from __future__ import annotations

import time
from pathlib import Path

from expflow.execution import (
    STATE_CANCELLED,
    STATE_COMPLETED,
    STATE_FAILED,
    STATE_QUEUED,
    STATE_RUNNING,
    STATE_UNKNOWN,
    LocalBackend,
    SlurmBackend,
    auto_detect_backend,
)

# ---------------------------------------------------------------------------
# LocalBackend
# ---------------------------------------------------------------------------


def _write_script(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(body)
    p.chmod(0o755)
    return p


def test_local_backend_runs_simple_script(tmp_path):
    backend = LocalBackend(tmp_path)
    script = _write_script(tmp_path, "ok.sh", "#!/bin/bash\necho hello\nexit 0\n")
    job_id = backend.submit(script)
    state = backend.wait(job_id, timeout=10)
    assert state == STATE_COMPLETED
    log = Path(backend.list_jobs()[job_id]["log_path"])
    assert "hello" in log.read_text()


def test_local_backend_failed_exit_code(tmp_path):
    backend = LocalBackend(tmp_path)
    script = _write_script(tmp_path, "fail.sh", "#!/bin/bash\necho oops\nexit 7\n")
    job_id = backend.submit(script)
    state = backend.wait(job_id, timeout=10)
    assert state == STATE_FAILED
    assert backend.list_jobs()[job_id]["exit_code"] == "7"


def test_local_backend_cancel(tmp_path):
    backend = LocalBackend(tmp_path, poll_interval=0.05)
    # Long-running script we can interrupt before it finishes.
    script = _write_script(tmp_path, "slow.sh", "#!/bin/bash\nsleep 30\n")
    job_id = backend.submit(script)
    # Wait until it transitions to RUNNING
    deadline = time.monotonic() + 5
    while backend.status(job_id) != STATE_RUNNING and time.monotonic() < deadline:
        time.sleep(0.05)
    assert backend.status(job_id) == STATE_RUNNING
    backend.cancel(job_id)
    state = backend.wait(job_id, timeout=10)
    assert state == STATE_CANCELLED


def test_local_backend_dependency_blocks_until_predecessor_completes(tmp_path):
    backend = LocalBackend(tmp_path, poll_interval=0.05)
    marker = tmp_path / "sentinel"

    pre = _write_script(
        tmp_path,
        "pre.sh",
        f"#!/bin/bash\nsleep 0.4\ntouch {marker}\nexit 0\n",
    )
    post = _write_script(
        tmp_path,
        "post.sh",
        f"#!/bin/bash\ntest -f {marker} && echo present || echo missing\n",
    )
    pre_id = backend.submit(pre)
    post_id = backend.submit(post, depends_on=[pre_id])

    state = backend.wait(post_id, timeout=10)
    assert state == STATE_COMPLETED
    log = Path(backend.list_jobs()[post_id]["log_path"]).read_text()
    assert "present" in log


def test_local_backend_dependency_propagates_failure(tmp_path):
    backend = LocalBackend(tmp_path, poll_interval=0.05)
    pre = _write_script(tmp_path, "pre.sh", "#!/bin/bash\nexit 1\n")
    post = _write_script(tmp_path, "post.sh", "#!/bin/bash\necho should_not_run\n")
    pre_id = backend.submit(pre)
    post_id = backend.submit(post, depends_on=[pre_id])
    state = backend.wait(post_id, timeout=10)
    assert state == STATE_CANCELLED


def test_local_backend_state_persisted_across_instances(tmp_path):
    a = LocalBackend(tmp_path)
    script = _write_script(tmp_path, "x.sh", "#!/bin/bash\nexit 0\n")
    job_id = a.submit(script)
    a.wait(job_id, timeout=10)

    b = LocalBackend(tmp_path)
    assert job_id in b.list_jobs()
    assert b.status(job_id) == STATE_COMPLETED


def test_local_backend_status_unknown_for_missing(tmp_path):
    backend = LocalBackend(tmp_path)
    assert backend.status("no-such-job") == STATE_UNKNOWN


def test_local_backend_passes_env_var(tmp_path):
    backend = LocalBackend(tmp_path)
    script = _write_script(
        tmp_path, "env.sh", "#!/bin/bash\necho FOO=$FOO\n"
    )
    job_id = backend.submit(script, env={"FOO": "bar123"})
    backend.wait(job_id, timeout=10)
    log = Path(backend.list_jobs()[job_id]["log_path"]).read_text()
    assert "FOO=bar123" in log


# ---------------------------------------------------------------------------
# SlurmBackend (no actual SLURM available; only static checks)
# ---------------------------------------------------------------------------


def test_slurm_backend_state_map_covers_common_states():
    bk = SlurmBackend(user="nobody")
    # Live states
    assert bk.SLURM_STATE_MAP["R"] == STATE_RUNNING
    assert bk.SLURM_STATE_MAP["RUNNING"] == STATE_RUNNING
    assert bk.SLURM_STATE_MAP["PD"] == STATE_QUEUED
    # Terminal states
    assert bk.SLURM_STATE_MAP["COMPLETED"] == STATE_COMPLETED
    assert bk.SLURM_STATE_MAP["FAILED"] == STATE_FAILED
    assert bk.SLURM_STATE_MAP["CANCELLED"] == STATE_CANCELLED
    assert bk.SLURM_STATE_MAP["TIMEOUT"] == STATE_FAILED
    assert bk.SLURM_STATE_MAP["OUT_OF_MEMORY"] == STATE_FAILED


def test_slurm_backend_user_falls_back_to_pwd(monkeypatch):
    monkeypatch.delenv("USER", raising=False)
    bk = SlurmBackend()
    # Whatever pwd returns, it should be a non-empty string
    assert isinstance(bk.user, str)


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------


def test_auto_detect_backend_returns_local_when_no_sbatch(tmp_path, monkeypatch):
    # Strip PATH so shutil.which("sbatch") returns None
    monkeypatch.setenv("PATH", "")
    backend = auto_detect_backend(tmp_path)
    assert backend.backend_name == "local"


def test_auto_detect_backend_returns_slurm_when_sbatch_on_path(tmp_path, monkeypatch):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch = fake_bin / "sbatch"
    sbatch.write_text("#!/bin/sh\necho fake\n")
    sbatch.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin))
    backend = auto_detect_backend(tmp_path)
    assert backend.backend_name == "slurm"


# ---------------------------------------------------------------------------
# Manager wiring (uses LocalBackend)
# ---------------------------------------------------------------------------


def _make_manager(project_root: Path):
    """Build a tiny manager subclass for integration tests."""
    from expflow.hpc_config import HPCConfig
    from expflow.hpcexp_core import BaseExperimentManager

    class M(BaseExperimentManager):
        def _generate_train_script(self, c):
            return f"#!/bin/bash\necho train {c.get('exp_id')}\nexit 0\n"

        def _generate_eval_script(self, c):
            return f"#!/bin/bash\necho eval {c.get('exp_id')}\nexit 0\n"

        def harvest_results(self, exp_id):
            return {}

    cfg = HPCConfig(
        username="u",
        user_home=str(project_root),
        scratch_dir=str(project_root),
        project_name="t",
        project_root=str(project_root),
        default_account="acct",
        default_partition="cpu",
    )
    return M(cfg, backend=LocalBackend(project_root, poll_interval=0.05))


def test_manager_submits_through_local_backend_end_to_end(tmp_path):
    mgr = _make_manager(tmp_path)
    import yaml
    (mgr.configs_dir / "demo.yaml").write_text(
        yaml.dump({"exp_id": "demo", "description": "x"})
    )
    mgr.register_experiments(["demo"])

    job_ids = mgr.submit_experiment("demo")
    assert "train_job_id" in job_ids
    assert "eval_job_id" in job_ids

    # Wait for both to finish
    mgr.backend.wait(job_ids["train_job_id"], timeout=10)
    state = mgr.backend.wait(job_ids["eval_job_id"], timeout=10)
    assert state == STATE_COMPLETED


def test_manager_get_slurm_jobs_routes_through_backend(tmp_path):
    mgr = _make_manager(tmp_path)
    # Empty before any submission
    assert mgr._get_slurm_jobs() == {}

    import yaml
    (mgr.configs_dir / "demo2.yaml").write_text(
        yaml.dump({"exp_id": "demo2", "description": "x"})
    )
    mgr.register_experiments(["demo2"])
    mgr.submit_experiment("demo2", train_only=True)

    jobs = mgr._get_slurm_jobs()
    assert len(jobs) >= 1
    a_job = next(iter(jobs.values()))
    assert "state" in a_job
