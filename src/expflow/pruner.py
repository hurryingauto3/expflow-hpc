#!/usr/bin/env python3
"""
ExpFlow Pruning System - Clean up duplicate and incomplete experiments

This module provides functionality to prune experiment directories:
- Remove duplicate runs (keep most recent only)
- Remove experiments without valid results or checkpoints
- Safe deletion (moves to .archive instead of permanent deletion)
"""

import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class PruneStats:
    """Statistics from pruning operation"""
    total_found: int = 0
    kept: int = 0
    pruned: int = 0
    space_freed_mb: float = 0.0
    duplicates_removed: int = 0
    invalid_removed: int = 0


class ExperimentPruner:
    """
    Prune experiment directories to clean up disk space

    Features:
    - Keep only most recent runs of duplicate experiments
    - Remove experiments without valid eval results or checkpoints
    - Safe deletion (move to .archive)
    - Dry-run mode for preview
    """

    def __init__(
        self,
        experiments_dir: Path,
        evaluations_dir: Optional[Path] = None,
        archive_dir: Optional[Path] = None
    ):
        """
        Initialize pruner

        Args:
            experiments_dir: Path to experiments/training directory
            evaluations_dir: Path to experiments/evaluations directory (optional)
            archive_dir: Path to archive directory (defaults to experiments_dir/.archive)
        """
        self.experiments_dir = Path(experiments_dir)
        self.evaluations_dir = Path(evaluations_dir) if evaluations_dir else None
        self.archive_dir = Path(archive_dir) if archive_dir else self.experiments_dir.parent / ".archive"

        # Ensure archive directory exists
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    def _extract_exp_base_name(self, exp_dir_name: str) -> Tuple[str, Optional[datetime]]:
        """
        Extract base experiment name and timestamp from directory name

        Examples:
            exp_a10_transfuser_agent_dinov2_refactor_100pct_20251127_100715
            -> ("exp_a10_transfuser_agent_dinov2_refactor_100pct", datetime(2025, 11, 27, 10, 7, 15))

            B13_20260124_014707
            -> ("B13", datetime(2026, 1, 24, 1, 47, 7))

        Args:
            exp_dir_name: Directory name

        Returns:
            (base_name, timestamp) tuple. timestamp is None if not parseable
        """
        # Try to extract timestamp from end of directory name
        # Pattern: YYYYMMDD_HHMMSS
        parts = exp_dir_name.split("_")

        # Look for date-like pattern in last two parts
        if len(parts) >= 2:
            date_part = parts[-2]
            time_part = parts[-1]

            # Check if they look like YYYYMMDD and HHMMSS
            if len(date_part) == 8 and date_part.isdigit() and len(time_part) == 6 and time_part.isdigit():
                try:
                    timestamp = datetime.strptime(f"{date_part}_{time_part}", "%Y%m%d_%H%M%S")
                    # Base name is everything except timestamp
                    base_name = "_".join(parts[:-2])
                    return (base_name, timestamp)
                except ValueError:
                    pass

        # Could not extract timestamp
        return (exp_dir_name, None)

    def _group_experiments_by_base_name(
        self,
        exp_dirs: List[Path]
    ) -> Dict[str, List[Tuple[Path, Optional[datetime]]]]:
        """
        Group experiment directories by base name

        Args:
            exp_dirs: List of experiment directory paths

        Returns:
            Dictionary mapping base_name -> [(path, timestamp), ...]
        """
        groups = defaultdict(list)

        for exp_dir in exp_dirs:
            base_name, timestamp = self._extract_exp_base_name(exp_dir.name)
            groups[base_name].append((exp_dir, timestamp))

        # Sort each group by timestamp (most recent first)
        for base_name in groups:
            groups[base_name].sort(
                key=lambda x: x[1] if x[1] is not None else datetime.min,
                reverse=True
            )

        return groups

    def _has_valid_checkpoint(self, exp_dir: Path, required_epochs: Optional[int] = None) -> bool:
        """
        Check if experiment has valid checkpoint(s)

        Args:
            exp_dir: Experiment directory
            required_epochs: If specified, check for checkpoint with this many epochs

        Returns:
            True if valid checkpoint exists
        """
        # Look for checkpoint files
        checkpoint_patterns = [
            "checkpoint*.pth",
            "checkpoint*.pt",
            "model*.pth",
            "model*.pt",
            "*.ckpt",
        ]

        for pattern in checkpoint_patterns:
            matches = list(exp_dir.rglob(pattern))
            if matches:
                # If no specific epoch requirement, any checkpoint is valid
                if required_epochs is None:
                    return True

                # Check if any checkpoint has required epochs
                # This would need to be customized based on checkpoint naming
                for ckpt in matches:
                    # Common patterns: checkpoint_epoch_50.pth, model_e100.pt, etc.
                    if f"epoch_{required_epochs}" in ckpt.name or f"e{required_epochs}" in ckpt.name:
                        return True

        return False

    def _has_valid_eval_results(self, exp_dir: Path, eval_dir: Optional[Path] = None) -> bool:
        """
        Check if experiment has valid evaluation results

        Args:
            exp_dir: Training experiment directory
            eval_dir: Optional evaluation directory to check

        Returns:
            True if valid eval results exist
        """
        # Check in training directory
        result_patterns = [
            "results.json",
            "metrics.json",
            "eval_results.json",
            "*_results.json",
            "metrics.txt",
            "results.txt",
        ]

        for pattern in result_patterns:
            matches = list(exp_dir.rglob(pattern))
            if matches:
                # Check that file is not empty
                for result_file in matches:
                    if result_file.stat().st_size > 0:
                        return True

        # Check evaluation directory if provided
        if eval_dir and eval_dir.exists():
            for pattern in result_patterns:
                matches = list(eval_dir.rglob(pattern))
                if matches:
                    for result_file in matches:
                        if result_file.stat().st_size > 0:
                            return True

        return False

    def _find_corresponding_eval_dir(self, train_dir: Path) -> Optional[Path]:
        """
        Find corresponding evaluation directory for a training directory

        Args:
            train_dir: Training directory path

        Returns:
            Evaluation directory path if found
        """
        if not self.evaluations_dir or not self.evaluations_dir.exists():
            return None

        train_name = train_dir.name

        # Common patterns for eval directories
        eval_patterns = [
            f"eval_*_{train_name}",
            f"eval_{train_name}",
            f"*_{train_name}",
        ]

        for pattern in eval_patterns:
            matches = list(self.evaluations_dir.glob(pattern))
            if matches:
                # Return most recent
                return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]

        return None

    def _get_dir_size_mb(self, directory: Path) -> float:
        """Get directory size in megabytes"""
        total = 0
        try:
            for item in directory.rglob("*"):
                if item.is_file():
                    total += item.stat().st_size
        except (PermissionError, FileNotFoundError):
            pass
        return total / (1024 * 1024)

    def _safe_delete(self, directory: Path, dry_run: bool = False) -> float:
        """
        Safely delete directory by moving to archive

        Args:
            directory: Directory to delete
            dry_run: If True, don't actually move

        Returns:
            Size freed in MB
        """
        size_mb = self._get_dir_size_mb(directory)

        if not dry_run:
            # Create timestamped archive subdirectory
            archive_subdir = self.archive_dir / datetime.now().strftime("%Y%m%d")
            archive_subdir.mkdir(parents=True, exist_ok=True)

            # Move directory to archive
            dest = archive_subdir / directory.name
            # If dest exists, append timestamp
            if dest.exists():
                dest = archive_subdir / f"{directory.name}_{datetime.now().strftime('%H%M%S')}"

            shutil.move(str(directory), str(dest))

        return size_mb

    def prune_duplicates(
        self,
        keep_n: int = 1,
        dry_run: bool = False,
        verbose: bool = True
    ) -> PruneStats:
        """
        Prune duplicate experiment runs, keeping only the N most recent

        Args:
            keep_n: Number of most recent runs to keep per experiment
            dry_run: If True, only show what would be deleted
            verbose: If True, print detailed information

        Returns:
            PruneStats with operation results
        """
        stats = PruneStats()

        if not self.experiments_dir.exists():
            if verbose:
                print(f"Experiments directory not found: {self.experiments_dir}")
            return stats

        # Get all experiment directories
        exp_dirs = [d for d in self.experiments_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        stats.total_found = len(exp_dirs)

        if verbose:
            print(f"\nScanning {stats.total_found} experiments in {self.experiments_dir}")

        # Group by base name
        groups = self._group_experiments_by_base_name(exp_dirs)

        if verbose:
            print(f"Found {len(groups)} unique experiment groups")

        # Process each group
        for base_name, runs in groups.items():
            if len(runs) <= keep_n:
                # No duplicates, keep all
                stats.kept += len(runs)
                continue

            # Keep first N (most recent)
            to_keep = runs[:keep_n]
            to_prune = runs[keep_n:]

            stats.kept += len(to_keep)
            stats.duplicates_removed += len(to_prune)

            if verbose:
                print(f"\n[{base_name}]")
                print(f"  Found {len(runs)} runs, keeping {keep_n} most recent")

            for exp_dir, timestamp in to_keep:
                if verbose:
                    ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S") if timestamp else "unknown"
                    print(f"    [KEEP] {exp_dir.name} ({ts_str})")

            for exp_dir, timestamp in to_prune:
                ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S") if timestamp else "unknown"
                size_mb = self._safe_delete(exp_dir, dry_run=dry_run)
                stats.space_freed_mb += size_mb
                stats.pruned += 1

                if verbose:
                    action = "[DRY-RUN]" if dry_run else "[PRUNE]"
                    print(f"    {action} {exp_dir.name} ({ts_str}) - {size_mb:.1f} MB")

        return stats

    def prune_invalid(
        self,
        require_checkpoint: bool = True,
        require_eval: bool = True,
        required_epochs: Optional[int] = None,
        dry_run: bool = False,
        verbose: bool = True
    ) -> PruneStats:
        """
        Prune experiments without valid checkpoints or evaluation results

        Args:
            require_checkpoint: If True, prune experiments without checkpoints
            require_eval: If True, prune experiments without eval results
            required_epochs: If specified, require checkpoint with this many epochs
            dry_run: If True, only show what would be deleted
            verbose: If True, print detailed information

        Returns:
            PruneStats with operation results
        """
        stats = PruneStats()

        if not self.experiments_dir.exists():
            if verbose:
                print(f"Experiments directory not found: {self.experiments_dir}")
            return stats

        # Get all experiment directories
        exp_dirs = [d for d in self.experiments_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        stats.total_found = len(exp_dirs)

        if verbose:
            print(f"\nValidating {stats.total_found} experiments...")
            if require_checkpoint:
                print(f"  Checking for valid checkpoints" + (f" (>= {required_epochs} epochs)" if required_epochs else ""))
            if require_eval:
                print(f"  Checking for evaluation results")

        for exp_dir in exp_dirs:
            valid = True
            reasons = []

            # Check checkpoint
            if require_checkpoint:
                if not self._has_valid_checkpoint(exp_dir, required_epochs):
                    valid = False
                    reasons.append("missing checkpoint")

            # Check eval results
            if require_eval:
                eval_dir = self._find_corresponding_eval_dir(exp_dir)
                if not self._has_valid_eval_results(exp_dir, eval_dir):
                    valid = False
                    reasons.append("missing eval results")

            if valid:
                stats.kept += 1
            else:
                stats.invalid_removed += 1
                stats.pruned += 1
                size_mb = self._safe_delete(exp_dir, dry_run=dry_run)
                stats.space_freed_mb += size_mb

                if verbose:
                    action = "[DRY-RUN]" if dry_run else "[PRUNE]"
                    reason_str = ", ".join(reasons)
                    print(f"  {action} {exp_dir.name} - {reason_str} - {size_mb:.1f} MB")

        if verbose and stats.kept > 0:
            print(f"\n  {stats.kept} valid experiments kept")

        return stats

    def prune_all(
        self,
        keep_n: int = 1,
        require_checkpoint: bool = True,
        require_eval: bool = True,
        required_epochs: Optional[int] = None,
        dry_run: bool = False,
        verbose: bool = True
    ) -> PruneStats:
        """
        Comprehensive pruning: remove duplicates and invalid experiments

        Args:
            keep_n: Number of most recent runs to keep per experiment
            require_checkpoint: If True, prune experiments without checkpoints
            require_eval: If True, prune experiments without eval results
            required_epochs: If specified, require checkpoint with this many epochs
            dry_run: If True, only show what would be deleted
            verbose: If True, print detailed information

        Returns:
            Combined PruneStats from both operations
        """
        if verbose:
            print("=" * 70)
            print("ExpFlow Experiment Pruner")
            print("=" * 70)
            mode = "[DRY RUN MODE]" if dry_run else "[LIVE MODE]"
            print(f"{mode}\n")

        # First, prune invalid experiments
        if verbose:
            print("\nStep 1: Removing invalid experiments...")
            print("-" * 70)

        stats_invalid = self.prune_invalid(
            require_checkpoint=require_checkpoint,
            require_eval=require_eval,
            required_epochs=required_epochs,
            dry_run=dry_run,
            verbose=verbose
        )

        # Then, prune duplicates among remaining experiments
        if verbose:
            print("\n\nStep 2: Removing duplicate runs...")
            print("-" * 70)

        stats_dupes = self.prune_duplicates(
            keep_n=keep_n,
            dry_run=dry_run,
            verbose=verbose
        )

        # Combine stats
        combined = PruneStats(
            total_found=stats_invalid.total_found,
            kept=stats_dupes.kept,  # Use final kept count from duplicate pruning
            pruned=stats_invalid.pruned + stats_dupes.pruned,
            space_freed_mb=stats_invalid.space_freed_mb + stats_dupes.space_freed_mb,
            duplicates_removed=stats_dupes.duplicates_removed,
            invalid_removed=stats_invalid.invalid_removed
        )

        # Print summary
        if verbose:
            print("\n" + "=" * 70)
            print("Summary")
            print("=" * 70)
            print(f"Total experiments scanned: {combined.total_found}")
            print(f"Kept: {combined.kept}")
            print(f"Pruned: {combined.pruned}")
            print(f"  - Invalid (missing results/checkpoints): {combined.invalid_removed}")
            print(f"  - Duplicates: {combined.duplicates_removed}")
            print(f"Space freed: {combined.space_freed_mb:.1f} MB ({combined.space_freed_mb / 1024:.2f} GB)")
            if dry_run:
                print("\n[DRY RUN] No files were actually deleted")
                print("Run without --dry-run to perform actual pruning")
            else:
                print(f"\nArchived to: {self.archive_dir}")
            print("=" * 70)

        return combined
