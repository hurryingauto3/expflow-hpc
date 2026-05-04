"""
Scope-based evaluation aggregation.

A "scope" is any categorical filter applied to evaluation: city, sub-task,
scenario tag, time bucket, dataset shard. ``BaseScopeAggregator`` discovers
per-scope evaluation directories, parses their CSVs, and produces a unified
record with weighted aggregation across scopes plus an "overall" track.

Extracted from navsim_manager.py:collect_results so any manager can
aggregate per-shard / per-city / per-scenario eval results without
re-implementing the discovery, grouping, and weighting logic.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Set


class BaseScopeAggregator(ABC):
    """
    Aggregate per-scope eval CSVs into a single record.

    Subclasses declare two things — ``known_scopes()`` (the set of scope
    labels the manager evaluates against) and ``column_map()`` (CSV column
    -> short metric name). The base implementation handles directory
    discovery, strict ID matching (no substring leaks), per-scope CSV
    parsing, and weighted aggregation.

    Example (NAVSIM):
        class CityAggregator(BaseScopeAggregator):
            def known_scopes(self):
                return {"boston", "vegas", "pittsburgh", "singapore", "all"}
            def column_map(self):
                return {"PDMS": "score", "NC": "no_at_fault_collisions", ...}

        agg = CityAggregator()
        result = agg.aggregate("exp_b21", evaluations_dir=Path("..."))
    """

    # ── Abstract hooks ──────────────────────────────────────────────────

    @abstractmethod
    def known_scopes(self) -> Set[str]:
        """Set of scope labels that may appear in eval-dir names."""

    @abstractmethod
    def column_map(self) -> Dict[str, str]:
        """Short metric name -> CSV column name. ``score`` is mandatory key."""

    # ── Tunable hooks (override if your conventions differ) ─────────────

    def primary_metric(self) -> str:
        """Short name of the headline metric (default: ``score``)."""
        return "score"

    def overall_scope(self) -> str:
        """Scope name for the "all-scopes-at-once" eval (default: ``all``)."""
        return "all"

    def filter_summary_rows(self, df: "Any") -> "Any":  # pragma: no cover - pandas dep
        """Hook to drop summary rows (e.g. NAVSIM's ``average_all_frames``)."""
        if "token" in df.columns:
            df = df[df["token"] != "average_all_frames"]
        return df

    # ── Public API ──────────────────────────────────────────────────────

    def discover_eval_dirs(self, evaluations_dir: Path, exp_id: str) -> List[Path]:
        """
        Find eval directories belonging to ``exp_id`` under ``evaluations_dir``.

        Uses strict prefix/suffix matching to avoid substring leaks
        (e.g. ``A3`` should not match ``A3-b``).
        """
        if not evaluations_dir.exists():
            return []
        out = []
        for d in evaluations_dir.glob(f"*{exp_id}*"):
            if not d.is_dir():
                continue
            name = d.name
            if (
                name == exp_id
                or name.startswith(f"{exp_id}_")
                or name.endswith(f"_{exp_id}")
                or f"_{exp_id}_" in name
            ):
                out.append(d)
        return out

    def detect_scope(self, dir_name: str) -> str:
        """
        Extract the scope label from an eval-directory name.

        Falls back to the ``overall_scope`` label when no known scope
        appears in the name.
        """
        lname = dir_name.lower()
        for scope in self.known_scopes():
            # require word-boundary-style match to avoid "all" matching "smallest"
            if re.search(rf"(?:^|[_-]){re.escape(scope.lower())}(?:$|[_-])", lname):
                return scope
        return self.overall_scope()

    def parse_csv(self, csv_path: Path) -> Dict[str, Any]:
        """
        Parse a single eval CSV into a flat metric dict.

        Optional dependency: pandas. Falls back to the stdlib ``csv``
        reader (slower but no extra dependency).
        """
        try:
            import pandas as pd

            df = pd.read_csv(csv_path)
            df = self.filter_summary_rows(df)
            if df.empty:
                return {"_scenarios": 0}
            metrics: Dict[str, Any] = {"_scenarios": int(len(df))}
            for short, col in self.column_map().items():
                if col in df.columns:
                    vals = df[col].dropna()
                    if len(vals) > 0:
                        metrics[short] = float(vals.mean())
            return metrics
        except ImportError:  # pragma: no cover - stdlib fallback
            import csv as _csv

            rows = []
            with open(csv_path, newline="") as fh:
                rdr = _csv.DictReader(fh)
                for r in rdr:
                    if r.get("token") == "average_all_frames":
                        continue
                    rows.append(r)
            if not rows:
                return {"_scenarios": 0}
            metrics = {"_scenarios": len(rows)}
            for short, col in self.column_map().items():
                vals = []
                for r in rows:
                    v = r.get(col)
                    try:
                        if v is not None and v != "":
                            vals.append(float(v))
                    except (TypeError, ValueError):
                        continue
                if vals:
                    metrics[short] = sum(vals) / len(vals)
            return metrics

    def aggregate(
        self,
        exp_id: str,
        evaluations_dir: Path,
    ) -> Dict[str, Any]:
        """
        Build a unified per-scope + overall record for ``exp_id``.

        Returns a dict shaped:
            {
              "exp_id": str,
              "scopes": {<scope>: {<short_metric>: float, "scenarios": int, ...}},
              "weighted_scores": {<short_metric>: float},
              "overall_scores": {<short_metric>: float} | None,
              "primary_score": float,           # weighted or overall, see logic below
              "total_scenarios": int,
            }

        Aggregation logic:
        - If an ``overall_scope`` directory exists, its metrics become
          ``overall_scores``.
        - Otherwise the per-scope means weighted by scenario count are
          used as ``weighted_scores``.
        - ``primary_score`` is the overall metric if present, else the
          weighted one.
        """
        eval_dirs = self.discover_eval_dirs(evaluations_dir, exp_id)
        scopes: Dict[str, Dict[str, Any]] = {}
        for d in eval_dirs:
            scope = self.detect_scope(d.name)
            csvs = sorted(d.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not csvs:
                continue
            metrics = self.parse_csv(csvs[0])
            scenarios = metrics.pop("_scenarios", 0)
            if scenarios <= 0:
                continue
            entry = scopes.setdefault(
                scope, {"scenarios": 0, "eval_dir": str(d.name)}
            )
            # Prefer the most-recent CSV per scope; in practice we just take
            # the first one encountered because ``discover_eval_dirs`` is
            # already mtime-stable per dir.
            if entry["scenarios"] == 0:
                entry.update(metrics)
                entry["scenarios"] = scenarios
                entry["eval_dir"] = str(d.name)

        if not scopes:
            return {
                "exp_id": exp_id,
                "scopes": {},
                "weighted_scores": {},
                "overall_scores": None,
                "primary_score": None,
                "total_scenarios": 0,
            }

        overall_label = self.overall_scope()
        overall = scopes.get(overall_label)
        per_scope = {k: v for k, v in scopes.items() if k != overall_label}

        weighted: Dict[str, float] = {}
        total = sum(s["scenarios"] for s in per_scope.values()) if per_scope else 0
        if total > 0:
            for short in self.column_map().keys():
                num = 0.0
                for s in per_scope.values():
                    if short in s:
                        num += s[short] * s["scenarios"]
                weighted[short] = num / total

        primary = self.primary_metric()
        if overall and overall.get(primary) is not None:
            primary_score = overall[primary]
        elif primary in weighted:
            primary_score = weighted[primary]
        else:
            primary_score = None

        return {
            "exp_id": exp_id,
            "scopes": scopes,
            "weighted_scores": weighted,
            "overall_scores": dict(overall) if overall else None,
            "primary_score": primary_score,
            "total_scenarios": (overall["scenarios"] if overall else total),
        }


class NullScopeAggregator(BaseScopeAggregator):
    """
    No-op aggregator used as a default for managers without per-scope eval.

    Treats every result as belonging to the single ``overall_scope``.
    """

    def known_scopes(self) -> Set[str]:
        return {self.overall_scope()}

    def column_map(self) -> Dict[str, str]:
        return {"score": "score"}
