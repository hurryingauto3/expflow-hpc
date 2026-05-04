"""
Structured result records for experiment storage.

``ResultRecordBuilder`` composes a deeply-nested experiment record from
config + harvested metrics + SLURM/git state, in named sections
(``core`` / ``slurm`` / ``git`` / arbitrary custom sections like
``architecture`` / ``training`` / ``evaluation``). ``BaseRecordEnricher``
is the hook for project-specific derived fields (e.g. NAVSIM's
architecture-name inference, generalisation-gap calculation).

Extracted from navsim_manager.py:store_comprehensive_results so that any
manager subclass can build the same shape of record without copy-pasting
the assembly code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional


class ResultRecordBuilder:
    """
    Fluent builder for structured experiment records.

    Each ``*_section`` method appends data under a named key; ``build()``
    returns the assembled dict. Calls are chainable.

    Example:
        record = (
            ResultRecordBuilder()
            .core_section("exp_001", "completed", created_at="2026-01-01T00:00:00")
            .slurm_section({"partition": "l40s", "num_gpus": 4}, {"train_job_id": "12345"})
            .git_section({"git_commit": "abc", "git_branch": "main", "git_dirty": False})
            .custom_section("training", {"epochs": 30, "batch_size": 32})
            .build()
        )
    """

    def __init__(self) -> None:
        self._record: Dict[str, Any] = {}

    def core_section(
        self,
        exp_id: str,
        status: str,
        *,
        created_at: Optional[str] = None,
        submitted_at: Optional[str] = None,
        completed_at: Optional[str] = None,
        **extra: Any,
    ) -> "ResultRecordBuilder":
        """Top-level identifying + lifecycle fields."""
        self._record.update(
            {
                "exp_id": exp_id,
                "status": status,
                "created_at": created_at,
                "submitted_at": submitted_at,
                "completed_at": completed_at,
                **extra,
            }
        )
        return self

    def slurm_section(
        self,
        config: Dict[str, Any],
        exp_meta: Dict[str, Any],
        *,
        keys: Optional[list] = None,
    ) -> "ResultRecordBuilder":
        """SLURM resource + job-id snapshot under the ``slurm`` key."""
        keys = keys or [
            "partition",
            "gpu_constraint",
            "num_gpus",
            "num_nodes",
            "cpus_per_task",
            "mem",
            "time_limit",
            "account",
        ]
        slurm = {k: config.get(k) for k in keys}
        slurm["train_job_id"] = exp_meta.get("train_job_id")
        slurm["eval_job_id"] = exp_meta.get("eval_job_id")
        slurm["eval_job_ids"] = exp_meta.get("eval_job_ids", [])
        self._record["slurm"] = slurm
        return self

    def git_section(self, config: Dict[str, Any]) -> "ResultRecordBuilder":
        """Git commit/branch/dirty snapshot under the ``git`` key."""
        self._record["git"] = {
            "commit": config.get("git_commit"),
            "branch": config.get("git_branch"),
            "dirty": config.get("git_dirty"),
        }
        return self

    def custom_section(self, name: str, data: Dict[str, Any]) -> "ResultRecordBuilder":
        """
        Attach an arbitrary nested dict under ``name``.

        Use for project-specific sections like ``architecture``,
        ``training``, ``evaluation``, ``phase``.
        """
        self._record[name] = data
        return self

    def merge(self, data: Dict[str, Any]) -> "ResultRecordBuilder":
        """Shallow-merge ``data`` into the top-level record."""
        self._record.update(data)
        return self

    def build(self) -> Dict[str, Any]:
        """
        Finalise the record; stamps ``stored_at`` and returns the dict.

        Safe to call multiple times — each call refreshes ``stored_at``
        and returns a copy so the builder can be reused.
        """
        result = dict(self._record)
        result["stored_at"] = datetime.utcnow().isoformat() + "Z"
        return result


class BaseRecordEnricher(ABC):
    """
    Hook for deriving higher-level fields from raw config + harvested data.

    A manager calls ``enricher.enrich(record, config, harvested)`` after
    the builder has produced the base record but before the record goes
    into the results storage. Typical derivations:

    - infer ``architecture.name`` / ``backbone.name`` from raw config keys,
    - compute ``effective_batch_size = num_gpus * batch_size * accumulate_grad_batches``,
    - compute ``generalization_gap_percent`` from per-scope eval scores,
    - tag a ``phase`` from the experiment-id prefix.

    Subclasses override ``enrich`` to add project-specific fields. Return
    the (possibly mutated) record.
    """

    @abstractmethod
    def enrich(
        self,
        record: Dict[str, Any],
        config: Dict[str, Any],
        harvested: Dict[str, Any],
    ) -> Dict[str, Any]:
        ...


class NoopRecordEnricher(BaseRecordEnricher):
    """Default enricher that returns the record unchanged."""

    def enrich(
        self,
        record: Dict[str, Any],
        config: Dict[str, Any],
        harvested: Dict[str, Any],
    ) -> Dict[str, Any]:
        return record
