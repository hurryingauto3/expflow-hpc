"""
ExpFlow Analysis Pipeline

Declarative YAML-driven pipeline for post-hoc analysis across experiments.
Resolves checkpoint globs, calls a user-provided handler for each experiment,
and exports a run summary.

Extracted from the pattern in scripts/visualize_trajectories.py +
scripts/viz_config.yaml + scripts/run_visualizations.slurm.
"""

import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import yaml

from .checkpoint_validator import CheckpointResolver, ValidationReport

# =============================================================================
# Data classes
# =============================================================================

@dataclass
class PipelineExperiment:
    """
    A single experiment entry inside an AnalysisPipeline config.

    resolved_checkpoint and resolved_exp_config are populated by validate().
    """
    name: str
    exp_config: str                                  # path to experiment YAML
    checkpoint: str                                  # path or glob pattern
    output_subdir: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
    resolved_checkpoint: Optional[str] = None
    resolved_exp_config: Optional[Dict[str, Any]] = None


@dataclass
class PipelineResult:
    """Result from running a single experiment through the pipeline handler."""
    name: str
    success: bool
    checkpoint_path: str
    output_dir: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineSummary:
    """Aggregated results from pipeline.run()."""
    pipeline_name: str
    total: int
    succeeded: int
    failed: int
    results: List[PipelineResult] = field(default_factory=list)
    run_at: str = ""
    output_dir: str = ""


# =============================================================================
# AnalysisPipeline
# =============================================================================

class AnalysisPipeline:
    """
    Declarative YAML-driven pipeline for post-hoc analysis across experiments.

    Usage::

        pipeline = AnalysisPipeline.from_config("viz_configs/backbone_shootout.yaml")

        # Validate all checkpoints and exp_configs exist
        report = pipeline.validate()

        # Run analysis with user-provided handler
        def my_handler(name, exp_config, checkpoint_path, output_dir, pipeline_context):
            agent = load_agent(exp_config, checkpoint_path)
            results = run_eval(agent)
            save_results(results, output_dir)
            return {"pdms": results["pdm_score"]}

        summary = pipeline.run(handler=my_handler)
        pipeline.export_summary(format="json")

    YAML Schema::

        pipeline_name: "backbone_comparison"
        experiments:
          - name: "I-JEPA (frozen)"
            exp_config: "/path/to/exp.yaml"
            checkpoint: "/path/to/checkpoints/*.ckpt"  # glob OK
            output_subdir: ""       # optional; derived from name if empty
            extra: {}               # arbitrary data forwarded to handler
        output_dir: "/scratch/user/outputs/backbone_comparison"
        save_individual: true
        save_grid: false
        tokens_file: null
        num_random_samples: 8
        extra_data: {}              # forwarded to handler as pipeline_context
    """

    def __init__(
        self,
        pipeline_name: str,
        experiments: List[PipelineExperiment],
        output_dir: Union[str, Path],
        save_individual: bool = True,
        save_grid: bool = False,
        tokens_file: Optional[str] = None,
        num_random_samples: int = 8,
        extra_data: Optional[Dict[str, Any]] = None,
    ):
        self.pipeline_name = pipeline_name
        self.experiments = experiments
        self.output_dir = Path(output_dir)
        self.save_individual = save_individual
        self.save_grid = save_grid
        self.tokens_file = tokens_file
        self.num_random_samples = num_random_samples
        self.extra_data: Dict[str, Any] = extra_data or {}
        self._summary: Optional[PipelineSummary] = None

    # -------------------------------------------------------------------------
    # Factory
    # -------------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> "AnalysisPipeline":
        """
        Load a pipeline from a YAML config file.

        Args:
            config_path: Path to the YAML config.

        Returns:
            AnalysisPipeline instance.

        Raises:
            ValueError on schema violations (missing required fields).
            FileNotFoundError if config_path does not exist.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Pipeline config not found: {config_path}")

        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}

        pipeline_name = raw.get("pipeline_name") or config_path.stem
        output_dir = raw.get("output_dir")
        if not output_dir:
            raise ValueError("'output_dir' is required in pipeline config")

        raw_exps = raw.get("experiments", [])
        if not raw_exps:
            raise ValueError("'experiments' list is required and must not be empty")

        experiments = []
        for i, entry in enumerate(raw_exps):
            if "name" not in entry:
                raise ValueError(f"experiments[{i}]: 'name' is required")
            if "exp_config" not in entry:
                raise ValueError(f"experiments[{i}] ('{entry['name']}'): 'exp_config' is required")
            if "checkpoint" not in entry:
                raise ValueError(f"experiments[{i}] ('{entry['name']}'): 'checkpoint' is required")
            experiments.append(PipelineExperiment(
                name=entry["name"],
                exp_config=str(entry["exp_config"]),
                checkpoint=str(entry["checkpoint"]),
                output_subdir=entry.get("output_subdir", ""),
                extra=entry.get("extra") or {},
            ))

        return cls(
            pipeline_name=pipeline_name,
            experiments=experiments,
            output_dir=output_dir,
            save_individual=raw.get("save_individual", True),
            save_grid=raw.get("save_grid", False),
            tokens_file=raw.get("tokens_file"),
            num_random_samples=raw.get("num_random_samples", 8),
            extra_data=raw.get("extra_data") or {},
        )

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate(self) -> ValidationReport:
        """
        Resolve all checkpoints and exp_config files. Cache results on each
        PipelineExperiment for use in run().

        Prints a summary of readiness.

        Returns:
            ValidationReport.
        """
        found = 0
        missing = []

        for exp in self.experiments:
            ok = True

            # Resolve checkpoint
            resolved_ckpt = CheckpointResolver.resolve(exp.checkpoint)
            if not resolved_ckpt:
                missing.append({
                    "name": exp.name,
                    "reason": "checkpoint not found",
                    "pattern": exp.checkpoint
                })
                ok = False
            else:
                exp.resolved_checkpoint = resolved_ckpt

            # Check exp_config
            if not Path(exp.exp_config).exists():
                missing.append({
                    "name": exp.name,
                    "reason": "exp_config not found",
                    "pattern": exp.exp_config
                })
                ok = False
            else:
                try:
                    with open(exp.exp_config) as f:
                        exp.resolved_exp_config = yaml.safe_load(f) or {}
                except Exception as e:
                    missing.append({
                        "name": exp.name,
                        "reason": f"failed to load exp_config: {e}",
                        "pattern": exp.exp_config
                    })
                    ok = False

            if ok:
                found += 1

        report = ValidationReport(
            found=found,
            total=len(self.experiments),
            missing=missing
        )
        print(str(report))
        return report

    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------

    def run(
        self,
        handler: Callable,
        output_dir: Optional[Union[str, Path]] = None,
        skip_invalid: bool = True,
        verbose: bool = True
    ) -> PipelineSummary:
        """
        Execute the pipeline by calling handler for each experiment.

        Handler signature::

            def handler(
                name: str,                        # experiment display name
                exp_config: Dict[str, Any],       # loaded experiment YAML dict
                checkpoint_path: str,             # resolved checkpoint path
                output_dir: Path,                 # per-experiment output dir
                pipeline_context: Dict[str, Any]  # pipeline-level data
            ) -> Optional[Dict[str, Any]]:        # optional metadata for summary
                ...

        pipeline_context contains:
            pipeline_name, tokens_file, save_individual, save_grid,
            num_random_samples, plus all extra_data keys.

        Per-experiment output_dir:
            pipeline.output_dir / slugified(output_subdir or name)

        Args:
            handler: User-provided callable with the signature above.
            output_dir: Override output directory (uses config value if None).
            skip_invalid: Skip experiments where checkpoint/config not resolved.
                          If False, raise on first failure.
            verbose: Print per-experiment progress.

        Returns:
            PipelineSummary.
        """
        effective_output = Path(output_dir) if output_dir else self.output_dir
        effective_output.mkdir(parents=True, exist_ok=True)

        context = {
            "pipeline_name": self.pipeline_name,
            "tokens_file": self.tokens_file,
            "save_individual": self.save_individual,
            "save_grid": self.save_grid,
            "num_random_samples": self.num_random_samples,
            **self.extra_data,
        }

        results: List[PipelineResult] = []

        for exp in self.experiments:
            if verbose:
                print(f"  [{self.experiments.index(exp) + 1}/{len(self.experiments)}] {exp.name}")

            # Ensure resolved — run lazy validation if not done yet
            if exp.resolved_checkpoint is None:
                exp.resolved_checkpoint = CheckpointResolver.resolve(exp.checkpoint)
            if exp.resolved_exp_config is None and Path(exp.exp_config).exists():
                with open(exp.exp_config) as f:
                    exp.resolved_exp_config = yaml.safe_load(f) or {}

            if exp.resolved_checkpoint is None or exp.resolved_exp_config is None:
                reason = (
                    "checkpoint not found" if exp.resolved_checkpoint is None
                    else "exp_config could not be loaded"
                )
                if not skip_invalid:
                    raise RuntimeError(f"Cannot run '{exp.name}': {reason}")
                if verbose:
                    print(f"    SKIP: {reason}")
                results.append(PipelineResult(
                    name=exp.name,
                    success=False,
                    checkpoint_path=exp.checkpoint,
                    output_dir="",
                    error=reason
                ))
                continue

            # Per-experiment output directory
            subdir = exp.output_subdir or exp.name
            exp_output = effective_output / _slugify(subdir)
            exp_output.mkdir(parents=True, exist_ok=True)

            try:
                handler_meta = handler(
                    name=exp.name,
                    exp_config=exp.resolved_exp_config,
                    checkpoint_path=exp.resolved_checkpoint,
                    output_dir=exp_output,
                    pipeline_context={**context, **exp.extra}
                )
                results.append(PipelineResult(
                    name=exp.name,
                    success=True,
                    checkpoint_path=exp.resolved_checkpoint,
                    output_dir=str(exp_output),
                    metadata=handler_meta or {}
                ))
                if verbose:
                    print("    [OK]")
            except Exception as e:
                results.append(PipelineResult(
                    name=exp.name,
                    success=False,
                    checkpoint_path=exp.resolved_checkpoint,
                    output_dir=str(exp_output),
                    error=str(e)
                ))
                if verbose:
                    print(f"    FAILED: {e}")

        succeeded = sum(1 for r in results if r.success)
        self._summary = PipelineSummary(
            pipeline_name=self.pipeline_name,
            total=len(self.experiments),
            succeeded=succeeded,
            failed=len(self.experiments) - succeeded,
            results=results,
            run_at=datetime.now().isoformat(),
            output_dir=str(effective_output)
        )

        print(f"\nPipeline '{self.pipeline_name}': "
              f"{succeeded}/{len(self.experiments)} succeeded")
        return self._summary

    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------

    def export_summary(
        self,
        output_path: Optional[Union[str, Path]] = None,
        format: str = "json"
    ) -> str:
        """
        Export pipeline run summary to a file.

        Args:
            output_path: Target file path. If None, writes to
                         {output_dir}/pipeline_summary.{format}.
            format: "json" or "csv".

        Returns:
            Path to the written file as string.

        Raises:
            RuntimeError if called before run().
        """
        if self._summary is None:
            raise RuntimeError("export_summary() must be called after run()")

        if output_path is None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.output_dir / f"pipeline_summary.{format}"

        output_path = Path(output_path)

        if format == "json":
            data = {
                "pipeline_name": self._summary.pipeline_name,
                "total": self._summary.total,
                "succeeded": self._summary.succeeded,
                "failed": self._summary.failed,
                "run_at": self._summary.run_at,
                "output_dir": self._summary.output_dir,
                "results": [
                    {
                        "name": r.name,
                        "success": r.success,
                        "checkpoint_path": r.checkpoint_path,
                        "output_dir": r.output_dir,
                        "error": r.error,
                        **r.metadata,
                    }
                    for r in self._summary.results
                ]
            }
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

        elif format == "csv":
            import csv
            fieldnames = ["name", "success", "checkpoint_path", "output_dir", "error"]
            # Collect extra metadata keys
            extra_keys: list = []
            for r in self._summary.results:
                for k in r.metadata:
                    if k not in extra_keys:
                        extra_keys.append(k)
            fieldnames.extend(extra_keys)

            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for r in self._summary.results:
                    row = {
                        "name": r.name,
                        "success": r.success,
                        "checkpoint_path": r.checkpoint_path,
                        "output_dir": r.output_dir,
                        "error": r.error or "",
                        **r.metadata,
                    }
                    writer.writerow(row)
        else:
            raise ValueError(f"Unsupported format: {format!r}. Use 'json' or 'csv'.")

        print(f"[OK] Summary written to {output_path}")
        return str(output_path)


# =============================================================================
# Internal helpers
# =============================================================================

def _slugify(text: str) -> str:
    """Convert a display name to a filesystem-safe directory name."""
    # Normalise unicode
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    # Replace non-alphanumeric with underscore
    text = re.sub(r"[^\w]+", "_", text)
    # Strip leading/trailing underscores
    return text.strip("_") or "experiment"
