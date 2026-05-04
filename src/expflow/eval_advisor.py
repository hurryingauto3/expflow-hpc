"""
Evaluation resource advisor.

Recommend ``num_workers`` / ``mem`` / ``time_limit`` for an evaluation
job based on the experiment config. The default implementation honours
explicit ``eval_workers`` / ``eval_mem`` / ``eval_time`` from the
experiment YAML and falls back to sane defaults; subclasses can plug
in a richer policy (e.g. NAVSIM's backbone-size scaling).

Extracted from navsim_manager.py:_generate_eval_script (the auto-scaling
block ~line 5400) so that any project can ship an advisor without
re-implementing the per-script scaling logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class EvalResourceAdvisor(ABC):
    """
    Recommend eval-time resource settings from an experiment config.

    A manager calls ``advisor.recommend(config)`` while generating the
    eval script and uses the returned dict for ``--cpus-per-task``,
    ``--mem``, and ``--time``.

    Returns a dict with at least these keys:
        - ``workers``: int
        - ``mem``: str (SLURM-format e.g. ``"160G"``)
        - ``time``: str (SLURM-format e.g. ``"06:00:00"``)
    """

    @abstractmethod
    def recommend(self, config: Dict[str, Any]) -> Dict[str, str]:
        ...


class DefaultEvalResourceAdvisor(EvalResourceAdvisor):
    """
    Honour explicit per-experiment overrides; fall back to sane defaults.

    Reads ``eval_workers`` / ``eval_mem`` / ``eval_time`` from the config
    if present, else defaults to 32 workers / 128G / 06:00:00. Memory is
    scaled with the worker count using ``mem_per_worker_gb`` (default 4)
    and clamped to ``mem_cap_gb`` (default 200) so the request is unlikely
    to be rejected on a 256 GB node.
    """

    def __init__(
        self,
        *,
        default_workers: int = 32,
        default_time: str = "06:00:00",
        mem_per_worker_gb: int = 4,
        mem_overhead_gb: int = 32,
        mem_cap_gb: int = 200,
    ) -> None:
        self.default_workers = default_workers
        self.default_time = default_time
        self.mem_per_worker_gb = mem_per_worker_gb
        self.mem_overhead_gb = mem_overhead_gb
        self.mem_cap_gb = mem_cap_gb

    def recommend(self, config: Dict[str, Any]) -> Dict[str, str]:
        workers = int(config.get("eval_workers") or self.default_workers)
        if "eval_mem" in config and config["eval_mem"]:
            mem = str(config["eval_mem"])
        else:
            scaled = workers * self.mem_per_worker_gb + self.mem_overhead_gb
            mem = f"{min(scaled, self.mem_cap_gb)}G"
        time_limit = str(config.get("eval_time") or self.default_time)
        return {"workers": workers, "mem": mem, "time": time_limit}
