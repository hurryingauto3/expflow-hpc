"""
Attempt-group bookkeeping for repeated evaluations.

When the same checkpoint is evaluated more than once (e.g. re-running
under a new evaluator version, or sweeping seeds at eval time), the run
records form an "attempt group". ``AttemptGrouping`` produces stable
hash-based group IDs from a configurable set of key fields, assigns
1-based ordering within each group, and materialises per-group summary
statistics.

Extracted from navsim_manager.py:
- _compute_eval_batch_id
- _compute_attempt_group_id
- _compute_eval_scope_key
- _assign_attempt_orders
- _build_attempt_summaries

so that any manager doing run-history analytics gets the bookkeeping
helpers without re-implementing them.
"""

from __future__ import annotations

import hashlib
import statistics
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


def _get_dotted(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Read ``path`` (dot-separated) from a nested dict; return ``default`` on miss."""
    cur: Any = d
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


class AttemptGrouping:
    """
    Compute attempt-group IDs and roll-up summaries from a list of run records.

    Args:
        key_fields: Dotted-path keys (against the run record) whose
                    values jointly identify an attempt group. Two runs
                    with identical values across all key fields belong
                    to the same group. Common choices:
                        ["checkpoint_path", "config.eval_type",
                         "config.train_split", "config.devkit_version"]
        primary_metric: Field name on each run used for best/mean/std
                        statistics in summaries (default: ``avg_pdms``;
                        also tries ``primary_score`` as a fallback).

    Example:
        g = AttemptGrouping(["checkpoint_path", "config.eval_type"])
        runs = [...]   # list of dicts loaded from run_results_storage
        g.assign_orders(runs)
        summaries = g.build_summaries(runs)
    """

    def __init__(
        self,
        key_fields: Iterable[str],
        *,
        primary_metric: str = "avg_pdms",
        timestamp_field: str = "attempt_timestamp",
    ) -> None:
        self.key_fields = list(key_fields)
        self.primary_metric = primary_metric
        self.timestamp_field = timestamp_field

    # ── Stable IDs ──────────────────────────────────────────────────────

    def group_id(self, exp_id: str, record: Dict[str, Any]) -> str:
        """Stable per-config attempt-group ID. Pure function of key_fields."""
        parts = [str(exp_id)]
        for path in self.key_fields:
            parts.append(str(_get_dotted(record, path) or ""))
        key = "|".join(parts)
        return "ag__" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def eval_batch_id(
        eval_job_ids: Optional[List[Any]] = None,
        timestamp: Optional[str] = None,
        slurm_job_id: Optional[str] = None,
    ) -> str:
        """
        ID for one multi-{scope|seed|...} evaluation batch.

        Uses sorted ``eval_job_ids`` when available (deterministic
        regardless of discovery order). Falls back to
        ``(timestamp, slurm_job_id)`` tuple string.
        """
        if eval_job_ids:
            return "batch__" + "_".join(sorted(str(j) for j in eval_job_ids))
        parts = [str(timestamp or ""), str(slurm_job_id or "")]
        return "batch__" + "_".join(p for p in parts if p)

    @staticmethod
    def scope_key(scope_results: Dict[str, Any]) -> str:
        """Sorted, normalised scope-set descriptor (for indexing/comparison)."""
        return ",".join(sorted(scope_results.keys()))

    # ── Ordering + summaries ────────────────────────────────────────────

    def assign_orders(self, runs: List[Dict[str, Any]]) -> None:
        """
        Mutate ``runs`` in place: set ``attempt_order`` (1-based) within
        each group, ordered by ``timestamp_field``.
        """
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in runs:
            ag = r.get("attempt_group_id")
            if ag:
                groups[ag].append(r)

        for members in groups.values():
            members.sort(key=lambda r: r.get(self.timestamp_field, ""))
            for idx, r in enumerate(members, start=1):
                r["attempt_order"] = idx

    def build_summaries(self, runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Build one summary per attempt group from the supplied runs.

        Returns a list of summary dicts (not stored anywhere — caller
        decides where to persist them). Each summary contains:
            attempt_group_id, exp_id, attempts_count,
            mean/std/min/max of primary_metric,
            latest_attempt_id, best_attempt_id, updated_at.
        """
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in runs:
            ag = r.get("attempt_group_id")
            if ag:
                groups[ag].append(r)

        summaries: List[Dict[str, Any]] = []
        for ag_id, members in groups.items():
            if not members:
                continue
            members.sort(key=lambda r: r.get(self.timestamp_field, ""))

            scores = [
                r.get(self.primary_metric)
                for r in members
                if r.get(self.primary_metric) is not None
            ]
            scores = [s for s in scores if s is not None and s > 0]

            latest = members[-1]
            best = max(
                members,
                key=lambda r: r.get(self.primary_metric) or 0.0,
            )

            summaries.append(
                {
                    "attempt_group_id": ag_id,
                    "exp_id": latest.get("exp_id"),
                    "attempts_count": len(members),
                    "mean_score": statistics.mean(scores) if scores else 0.0,
                    "std_score": (
                        statistics.stdev(scores) if len(scores) >= 2 else 0.0
                    ),
                    "min_score": min(scores) if scores else 0.0,
                    "max_score": max(scores) if scores else 0.0,
                    "latest_attempt_id": latest.get("run_id"),
                    "best_attempt_id": best.get("run_id"),
                    "updated_at": datetime.utcnow().isoformat() + "Z",
                }
            )
        return summaries
