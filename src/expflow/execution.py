"""
Execution backends for ExpFlow.

A backend abstracts away the difference between submitting an
experiment to a SLURM cluster and running it directly on the local
machine. Managers receive a backend in their constructor (auto-selected
if not supplied) and call its protocol methods instead of running
``sbatch`` / ``squeue`` / ``scancel`` directly.

Two backends ship with ExpFlow:

- ``LocalBackend``: laptop / CI / no-SLURM. Spawns scripts as detached
  ``bash`` subprocesses, persists state to ``project_root / ".local_jobs.json"``,
  honours ``depends_on`` by waiting for the predecessor job to finish in
  a background thread before launching the dependent.
- ``SlurmBackend``: thin wrapper around ``sbatch`` / ``squeue`` /
  ``scancel`` extracted from the legacy inline subprocess calls in
  ``hpcexp_core.py``.

Usage:

    from expflow.execution import LocalBackend
    mgr = MyManager(hpc_config, backend=LocalBackend(project_root))
    mgr.submit_experiment("exp1")     # runs in a local subprocess
    mgr.list_experiments()             # backend.list_jobs() merged in

If ``backend=None`` is passed, ``BaseExperimentManager`` auto-selects
``SlurmBackend`` when ``sbatch`` is on ``$PATH``, else ``LocalBackend``.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Protocol, runtime_checkable

# Job-state vocabulary (lowercase, normalised across backends)
STATE_QUEUED = "queued"
STATE_RUNNING = "running"
STATE_COMPLETED = "completed"
STATE_FAILED = "failed"
STATE_CANCELLED = "cancelled"
STATE_UNKNOWN = "unknown"

TERMINAL_STATES = {STATE_COMPLETED, STATE_FAILED, STATE_CANCELLED}


@runtime_checkable
class ExecutionBackend(Protocol):
    """
    Job execution protocol. Implementations must be importable on a
    login node without SLURM available — checks for ``sbatch`` belong
    inside the implementation, not at import time.
    """

    backend_name: str

    def submit(
        self,
        script_path: Path,
        *,
        depends_on: Optional[List[str]] = None,
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> str:
        """Schedule ``script_path`` to run; return an opaque job ID."""

    def cancel(self, job_id: str) -> None:
        """Cancel a queued or running job."""

    def status(self, job_id: str) -> str:
        """Return the current state of ``job_id`` (one of the constants above)."""

    def list_jobs(self) -> Dict[str, Dict[str, str]]:
        """Return ``{job_id: {state, submit_time, ...}}`` for all known jobs."""


# =============================================================================
# LocalBackend
# =============================================================================


@dataclass
class _LocalJob:
    job_id: str
    pid: Optional[int]
    script: str
    state: str
    submit_time: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    exit_code: Optional[int] = None
    log_path: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)


class LocalBackend:
    """
    Run scripts as local subprocesses.

    Job state is persisted in ``state_dir / ".local_jobs.json"`` so it
    survives between manager invocations. Dependencies are honoured by
    spawning a watcher thread that blocks until the predecessor reaches
    a terminal state before launching the dependent script.

    Args:
        state_dir: Directory where ``.local_jobs.json`` lives. Typical
                   choice: ``BaseExperimentManager.project_root``.
        log_dir:   Directory for ``{job_id}.out`` log files. Defaults to
                   ``state_dir / "logs"``.
        poll_interval: Seconds between dependency-wait poll cycles
                       (default 2).
    """

    backend_name = "local"
    _STATE_FILE = ".local_jobs.json"

    def __init__(
        self,
        state_dir: Path,
        *,
        log_dir: Optional[Path] = None,
        poll_interval: float = 2.0,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_dir) if log_dir else self.state_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.poll_interval = poll_interval
        self._lock = threading.Lock()
        # Map of in-flight watcher threads keyed by job_id, kept so the
        # process doesn't get GC'd before the script starts.
        self._watchers: Dict[str, threading.Thread] = {}

    # ── State persistence ──────────────────────────────────────────────

    def _load(self) -> Dict[str, _LocalJob]:
        path = self.state_dir / self._STATE_FILE
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        out: Dict[str, _LocalJob] = {}
        for jid, payload in raw.items():
            try:
                out[jid] = _LocalJob(**payload)
            except TypeError:
                continue
        return out

    def _save(self, jobs: Dict[str, _LocalJob]) -> None:
        path = self.state_dir / self._STATE_FILE
        tmp = path.with_suffix(".json.tmp")
        payload = {jid: job.__dict__ for jid, job in jobs.items()}
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp.replace(path)

    def _update(self, job_id: str, **fields: object) -> None:
        with self._lock:
            jobs = self._load()
            if job_id not in jobs:
                return
            for k, v in fields.items():
                setattr(jobs[job_id], k, v)
            self._save(jobs)

    # ── ExecutionBackend protocol ──────────────────────────────────────

    def submit(
        self,
        script_path: Path,
        *,
        depends_on: Optional[List[str]] = None,
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> str:
        script_path = Path(script_path)
        if not script_path.exists():
            raise FileNotFoundError(f"script not found: {script_path}")

        # UUID suffix prevents collision when two submit() calls land in
        # the same millisecond (the time-based prefix is kept only for
        # human-friendly chronological sorting in list_jobs output).
        job_id = f"local-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
        log_path = self.log_dir / f"{job_id}.out"

        with self._lock:
            jobs = self._load()
            jobs[job_id] = _LocalJob(
                job_id=job_id,
                pid=None,
                script=str(script_path),
                state=STATE_QUEUED,
                submit_time=time.strftime("%Y-%m-%dT%H:%M:%S"),
                log_path=str(log_path),
                depends_on=list(depends_on or []),
            )
            self._save(jobs)

        thread = threading.Thread(
            target=self._run_job,
            args=(job_id, script_path, depends_on or [], cwd, env, log_path),
            name=f"localbackend-{job_id}",
            daemon=True,
        )
        thread.start()
        self._watchers[job_id] = thread
        return job_id

    def _run_job(
        self,
        job_id: str,
        script_path: Path,
        depends_on: List[str],
        cwd: Optional[Path],
        env: Optional[Dict[str, str]],
        log_path: Path,
    ) -> None:
        # Wait for predecessors
        for dep in depends_on:
            while True:
                state = self.status(dep)
                if state in TERMINAL_STATES:
                    if state != STATE_COMPLETED:
                        # Predecessor failed/cancelled — propagate.
                        self._update(
                            job_id,
                            state=STATE_CANCELLED,
                            end_time=time.strftime("%Y-%m-%dT%H:%M:%S"),
                            exit_code=-1,
                        )
                        return
                    break
                time.sleep(self.poll_interval)

        # Launch the script
        proc_env = os.environ.copy()
        if env:
            proc_env.update({k: str(v) for k, v in env.items()})
        proc_env.setdefault("EXPFLOW_BACKEND_NAME", self.backend_name)
        proc_env.setdefault("EXPFLOW_LOCAL_JOB_ID", job_id)

        with open(log_path, "w") as log_fh:
            try:
                proc = subprocess.Popen(
                    ["bash", str(script_path)],
                    cwd=str(cwd) if cwd else None,
                    env=proc_env,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except OSError as e:
                self._update(
                    job_id,
                    state=STATE_FAILED,
                    end_time=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    exit_code=-1,
                )
                with open(log_path, "a") as log_fh2:
                    log_fh2.write(f"\n[localbackend] failed to spawn: {e}\n")
                return

            self._update(
                job_id,
                pid=proc.pid,
                state=STATE_RUNNING,
                start_time=time.strftime("%Y-%m-%dT%H:%M:%S"),
            )

            try:
                rc = proc.wait()
            except KeyboardInterrupt:
                proc.kill()
                rc = -signal.SIGINT

        end_state = STATE_COMPLETED if rc == 0 else STATE_FAILED
        # If we previously marked it cancelled (via cancel()), preserve that.
        with self._lock:
            jobs = self._load()
            current = jobs.get(job_id)
            if current and current.state == STATE_CANCELLED:
                end_state = STATE_CANCELLED

        self._update(
            job_id,
            state=end_state,
            end_time=time.strftime("%Y-%m-%dT%H:%M:%S"),
            exit_code=rc,
        )

    def cancel(self, job_id: str) -> None:
        with self._lock:
            jobs = self._load()
            job = jobs.get(job_id)
            if not job:
                return
            if job.state in TERMINAL_STATES:
                return
            pid = job.pid

        if pid:
            try:
                # Kill the entire process group (start_new_session=True above)
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass
        self._update(
            job_id,
            state=STATE_CANCELLED,
            end_time=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

    def status(self, job_id: str) -> str:
        jobs = self._load()
        job = jobs.get(job_id)
        if not job:
            return STATE_UNKNOWN
        # Detect crashed-without-exit-status processes by probing the PID.
        if job.state == STATE_RUNNING and job.pid:
            if not _pid_alive(job.pid):
                # Process died but watcher hasn't recorded it yet —
                # report running until the watcher catches up. This
                # avoids races on the very-rare PID-reuse window.
                pass
        return job.state

    def list_jobs(self) -> Dict[str, Dict[str, str]]:
        return {
            jid: {
                "state": job.state,
                "submit_time": job.submit_time,
                "start_time": job.start_time or "",
                "end_time": job.end_time or "",
                "pid": str(job.pid) if job.pid else "",
                "log_path": job.log_path or "",
                "exit_code": str(job.exit_code) if job.exit_code is not None else "",
            }
            for jid, job in self._load().items()
        }

    def wait(self, job_id: str, *, timeout: Optional[float] = None) -> str:
        """Block until ``job_id`` reaches a terminal state. Test helper."""
        deadline = time.monotonic() + timeout if timeout else None
        while True:
            state = self.status(job_id)
            if state in TERMINAL_STATES or state == STATE_UNKNOWN:
                return state
            if deadline is not None and time.monotonic() >= deadline:
                return state
            time.sleep(min(self.poll_interval, 0.2))


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


# =============================================================================
# SlurmBackend
# =============================================================================


class SlurmBackend:
    """
    SLURM-backed execution. Thin wrapper around ``sbatch`` / ``squeue`` /
    ``scancel``.

    Args:
        user:    User to filter ``squeue`` calls by. Defaults to
                 ``$USER`` or ``pwd.getpwuid(os.getuid()).pw_name``.
        timeout: Per-call subprocess timeout in seconds (default 30).
    """

    backend_name = "slurm"

    SLURM_STATE_MAP = {
        "PD": STATE_QUEUED,
        "CF": STATE_QUEUED,
        "PENDING": STATE_QUEUED,
        "CONFIGURING": STATE_QUEUED,
        "R": STATE_RUNNING,
        "RUNNING": STATE_RUNNING,
        "CG": STATE_RUNNING,
        "COMPLETING": STATE_RUNNING,
        "CD": STATE_COMPLETED,
        "COMPLETED": STATE_COMPLETED,
        "F": STATE_FAILED,
        "FAILED": STATE_FAILED,
        "TO": STATE_FAILED,
        "TIMEOUT": STATE_FAILED,
        "OOM": STATE_FAILED,
        "OUT_OF_MEMORY": STATE_FAILED,
        "BF": STATE_FAILED,
        "BOOT_FAIL": STATE_FAILED,
        "DL": STATE_FAILED,
        "DEADLINE": STATE_FAILED,
        "NF": STATE_FAILED,
        "NODE_FAIL": STATE_FAILED,
        "PR": STATE_FAILED,
        "PREEMPTED": STATE_FAILED,
        "CA": STATE_CANCELLED,
        "CANCELLED": STATE_CANCELLED,
        "S": STATE_QUEUED,
        "SUSPENDED": STATE_QUEUED,
    }

    def __init__(
        self,
        user: Optional[str] = None,
        *,
        timeout: int = 30,
    ) -> None:
        if user is None:
            user = os.environ.get("USER") or _detect_user()
        self.user = user
        self.timeout = timeout

    def submit(
        self,
        script_path: Path,
        *,
        depends_on: Optional[List[str]] = None,
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> str:
        cmd = ["sbatch"]
        if depends_on:
            cmd.extend(["--dependency", "afterok:" + ":".join(str(d) for d in depends_on)])
        cmd.append(str(script_path))

        proc_env = os.environ.copy()
        if env:
            proc_env.update({k: str(v) for k, v in env.items()})

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=str(cwd) if cwd else None,
            env=proc_env,
            timeout=self.timeout,
        )
        # sbatch output: "Submitted batch job 12345"
        return result.stdout.strip().split()[-1]

    def cancel(self, job_id: str) -> None:
        try:
            subprocess.run(
                ["scancel", str(job_id)],
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            pass

    def status(self, job_id: str) -> str:
        # Try squeue first (live state); fall back to sacct for terminal states.
        try:
            r = subprocess.run(
                ["squeue", "-j", str(job_id), "-h", "-o", "%T"],
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout,
            )
            state = r.stdout.strip().splitlines()[0] if r.stdout.strip() else ""
        except subprocess.TimeoutExpired:
            return STATE_UNKNOWN

        if state:
            return self.SLURM_STATE_MAP.get(state.upper(), STATE_UNKNOWN)

        try:
            r = subprocess.run(
                ["sacct", "-j", str(job_id), "-n", "-X", "-P", "-o", "State"],
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout,
            )
            line = r.stdout.strip().splitlines()[0] if r.stdout.strip() else ""
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return STATE_UNKNOWN
        if not line:
            return STATE_UNKNOWN
        return self.SLURM_STATE_MAP.get(line.upper().split()[0], STATE_UNKNOWN)

    def list_jobs(self) -> Dict[str, Dict[str, str]]:
        if not self.user:
            return {}
        try:
            result = subprocess.run(
                [
                    "squeue",
                    "-u",
                    self.user,
                    "-h",
                    "-o",
                    "%i|%T|%j|%P|%M|%l",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return {}

        out: Dict[str, Dict[str, str]] = {}
        for line in result.stdout.strip().splitlines():
            parts = line.split("|")
            if len(parts) < 6:
                continue
            jid, state, name, partition, runtime, time_limit = parts[:6]
            out[jid] = {
                "state": self.SLURM_STATE_MAP.get(state.upper(), state.lower()),
                "name": name,
                "partition": partition,
                "runtime": runtime,
                "time_limit": time_limit,
                "raw_state": state,
            }
        return out


# =============================================================================
# Auto-detection
# =============================================================================


def _detect_user() -> str:
    try:
        import pwd

        return pwd.getpwuid(os.getuid()).pw_name
    except Exception:
        return ""


def auto_detect_backend(state_dir: Path) -> ExecutionBackend:
    """
    Pick a backend automatically.

    Returns ``SlurmBackend`` when ``sbatch`` is on ``PATH``, else
    ``LocalBackend(state_dir)``. The check is cheap (``shutil.which``)
    so ``BaseExperimentManager.__init__`` can call this on every
    instantiation.
    """
    if shutil.which("sbatch"):
        return SlurmBackend()
    return LocalBackend(state_dir)
