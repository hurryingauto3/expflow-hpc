"""
ExpFlow Checkpoint Resolver and Validator

Utilities for resolving glob patterns to concrete checkpoint paths and
validating that all required artifacts exist before running analysis jobs.
"""

import glob as _glob
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from .hpcexp_core import BaseExperimentManager


# =============================================================================
# CheckpointResolver — standalone, no manager dependency
# =============================================================================

class CheckpointResolver:
    """
    Resolve glob patterns to concrete checkpoint file paths.

    Stateless utility — all methods are static and have no dependency on
    BaseExperimentManager. Safe to import on login nodes without pulling in
    the full framework.
    """

    STRATEGIES = ("mtime", "name_epoch", "name_step")

    @staticmethod
    def resolve(pattern: str, strategy: str = "mtime") -> Optional[str]:
        """
        Resolve a path or glob pattern to a single checkpoint file.

        Args:
            pattern: Direct file path or glob pattern. If no wildcards,
                     returns the path as-is if it exists.
            strategy: Selection strategy when multiple matches:
                      "mtime"       — most recently modified file (default)
                      "name_epoch"  — highest epoch number in filename
                      "name_step"   — highest step number in filename

        Returns:
            Absolute path string, or None if no match found.
        """
        pattern = str(pattern)

        # No wildcards — direct path check
        if "*" not in pattern and "?" not in pattern:
            return pattern if Path(pattern).is_file() else None

        matches = _glob.glob(pattern, recursive=True)
        if not matches:
            return None

        if strategy == "mtime":
            return max(matches, key=lambda p: Path(p).stat().st_mtime)
        elif strategy == "name_epoch":
            return max(matches, key=CheckpointResolver._extract_epoch)
        elif strategy == "name_step":
            return max(matches, key=CheckpointResolver._extract_step)
        else:
            return max(matches, key=lambda p: Path(p).stat().st_mtime)

    @staticmethod
    def resolve_all(pattern: str) -> List[str]:
        """Return all matches for a glob pattern, sorted by mtime descending."""
        pattern = str(pattern)
        if "*" not in pattern and "?" not in pattern:
            return [pattern] if Path(pattern).is_file() else []
        matches = _glob.glob(pattern, recursive=True)
        return sorted(matches, key=lambda p: Path(p).stat().st_mtime, reverse=True)

    @staticmethod
    def exists(pattern: str) -> bool:
        """Return True if the pattern resolves to at least one existing file."""
        return CheckpointResolver.resolve(pattern) is not None

    @staticmethod
    def _extract_epoch(path: str) -> int:
        """Extract epoch number from filename for sorting. Returns -1 if not found."""
        name = Path(path).stem
        for pat in (r"epoch[=_-](\d+)", r"epoch(\d+)", r"ep(\d+)"):
            m = re.search(pat, name, re.IGNORECASE)
            if m:
                return int(m.group(1))
        return -1

    @staticmethod
    def _extract_step(path: str) -> int:
        """Extract step number from filename for sorting. Returns -1 if not found."""
        name = Path(path).stem
        for pat in (r"step[=_-](\d+)", r"step(\d+)"):
            m = re.search(pat, name, re.IGNORECASE)
            if m:
                return int(m.group(1))
        return -1


# =============================================================================
# ValidationReport
# =============================================================================

@dataclass
class ValidationReport:
    """Result from CheckpointValidator methods."""
    found: int
    total: int
    missing: List[Dict[str, str]] = field(default_factory=list)
    # Each entry: {"name": str, "reason": str, "pattern": str}
    ready_to_run: bool = False
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.ready_to_run = (self.found == self.total)

    @property
    def missing_count(self) -> int:
        return self.total - self.found

    def __str__(self) -> str:
        status = "[READY]" if self.ready_to_run else "WARNING: not ready"
        lines = [f"ValidationReport: {self.found}/{self.total} ready  {status}"]
        for item in self.missing:
            lines.append(f"  MISSING {item.get('name', '?')}: {item.get('reason', '')}  "
                         f"(pattern: {item.get('pattern', '?')})")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


# =============================================================================
# CheckpointValidator
# =============================================================================

class CheckpointValidator:
    """
    Validate checkpoint files and experiment artifacts against expected configurations.

    Works with:
    - Standalone visualization config YAMLs (viz_config.yaml schema)
    - Manager-registered experiments (requires manager argument)
    """

    def __init__(self, manager: Optional["BaseExperimentManager"] = None):
        """
        Args:
            manager: Optional BaseExperimentManager instance for exp-aware validation.
                     Required for validate_experiments(). Not needed for validate_config().
        """
        self.manager = manager

    def validate_config(
        self,
        config_path: Union[str, Path],
        required_fields: Optional[List[str]] = None
    ) -> ValidationReport:
        """
        Validate all checkpoints referenced in a visualization config YAML.

        Expected YAML schema (matches viz_config.yaml):
          experiments:
            - name: "..."
              exp_config: "/path/to/exp.yaml"   # optional
              checkpoint: "/path/or/glob/*.ckpt" # required (or other required_fields)

        Args:
            config_path: Path to the YAML config file.
            required_fields: Per-experiment fields to verify. Defaults to ["checkpoint"].

        Returns:
            ValidationReport.
        """
        if required_fields is None:
            required_fields = ["checkpoint"]

        config_path = Path(config_path)
        if not config_path.exists():
            return ValidationReport(
                found=0, total=0,
                warnings=[f"Config file not found: {config_path}"]
            )

        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

        experiments = config.get("experiments", [])
        if not experiments:
            return ValidationReport(found=0, total=0, warnings=["No experiments listed in config"])

        found = 0
        missing = []

        for exp in experiments:
            name = exp.get("name", "?")
            ok = True

            for field_name in required_fields:
                val = exp.get(field_name)
                if not val:
                    missing.append({"name": name, "reason": f"field '{field_name}' missing", "pattern": ""})
                    ok = False
                    continue

                if field_name == "checkpoint":
                    resolved = CheckpointResolver.resolve(str(val))
                    if not resolved:
                        missing.append({"name": name, "reason": "checkpoint not found", "pattern": str(val)})
                        ok = False
                elif field_name == "exp_config":
                    if not Path(str(val)).exists():
                        missing.append({"name": name, "reason": "exp_config not found", "pattern": str(val)})
                        ok = False
                else:
                    # Generic file existence check
                    if not Path(str(val)).exists():
                        missing.append({"name": name, "reason": f"'{field_name}' not found", "pattern": str(val)})
                        ok = False

            if ok:
                found += 1

        report = ValidationReport(found=found, total=len(experiments), missing=missing)
        print(str(report))
        return report

    def validate_experiments(
        self,
        exp_ids: List[str],
        required_artifacts: Optional[List[str]] = None
    ) -> ValidationReport:
        """
        Validate artifacts for manager-registered experiments.

        Args:
            exp_ids: Experiment IDs to check.
            required_artifacts: Artifact types to verify. Options:
                "checkpoint"   — latest checkpoint file exists
                "results"      — results dict in metadata is non-empty
                "yaml"         — YAML config file exists
                "train_script" — generated train script exists
                "eval_script"  — generated eval script exists
                Defaults to ["checkpoint", "yaml"].

        Returns:
            ValidationReport.

        Raises:
            RuntimeError if no manager was provided.
        """
        if self.manager is None:
            raise RuntimeError("validate_experiments() requires a manager instance")

        if required_artifacts is None:
            required_artifacts = ["checkpoint", "yaml"]

        found = 0
        missing = []

        for exp_id in exp_ids:
            if exp_id not in self.manager.metadata:
                missing.append({"name": exp_id, "reason": "not registered in metadata", "pattern": ""})
                continue

            meta = self.manager.metadata[exp_id]
            ok = True

            for artifact in required_artifacts:
                if artifact == "yaml":
                    yaml_path = self.manager.configs_dir / f"{exp_id}.yaml"
                    if not yaml_path.exists():
                        missing.append({"name": exp_id, "reason": "YAML config missing", "pattern": str(yaml_path)})
                        ok = False

                elif artifact == "checkpoint":
                    ckpt = self.manager._find_latest_checkpoint(exp_id)
                    if not ckpt:
                        missing.append({"name": exp_id, "reason": "no checkpoint found",
                                        "pattern": str(self.manager.checkpoints_dir / exp_id / "*")})
                        ok = False

                elif artifact == "results":
                    if not meta.get("results"):
                        missing.append({"name": exp_id, "reason": "results empty", "pattern": ""})
                        ok = False

                elif artifact == "train_script":
                    path = meta.get("train_script_path")
                    if not path or not Path(path).exists():
                        missing.append({"name": exp_id, "reason": "train script missing",
                                        "pattern": path or ""})
                        ok = False

                elif artifact == "eval_script":
                    path = meta.get("eval_script_path")
                    if not path or not Path(path).exists():
                        missing.append({"name": exp_id, "reason": "eval script missing",
                                        "pattern": path or ""})
                        ok = False

            if ok:
                found += 1

        report = ValidationReport(found=found, total=len(exp_ids), missing=missing)
        print(str(report))
        return report

    def validate_directory(
        self,
        config_dir: Union[str, Path],
        priority_first: Optional[List[str]] = None
    ) -> Dict[str, ValidationReport]:
        """
        Validate all YAML configs in a directory.

        Args:
            config_dir: Directory containing viz config YAMLs.
            priority_first: Filenames (without path) to process first.

        Returns:
            Dict mapping filename -> ValidationReport.
        """
        config_dir = Path(config_dir)
        if not config_dir.exists():
            return {}

        yaml_files = sorted(config_dir.glob("*.yaml"))
        if priority_first:
            priority_set = set(priority_first)
            yaml_files = (
                [f for f in yaml_files if f.name in priority_set] +
                [f for f in yaml_files if f.name not in priority_set]
            )

        results = {}
        for yaml_path in yaml_files:
            results[yaml_path.name] = self.validate_config(yaml_path)

        total = len(results)
        ready = sum(1 for r in results.values() if r.ready_to_run)
        print(f"Directory validation: {ready}/{total} configs ready")
        return results
