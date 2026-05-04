"""
ExpFlow Experiment Series Builder

Generate batches of experiment configurations from a base config and a
parameter grid (Cartesian product), with optional checkpoint registry support.

Extracted from the pattern in scripts/generate_custom_vit_experiments.py.
"""

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import yaml

if TYPE_CHECKING:
    from .hpcexp_core import BaseExperimentManager

from .checkpoint_validator import CheckpointResolver

# =============================================================================
# Data classes
# =============================================================================

@dataclass
class CheckpointSpec:
    """
    Checkpoint descriptor for a single backbone/variant in a registry.

    path_field and key_field specify the YAML config keys used to inject the
    checkpoint path and key into each generated experiment config.
    If left as None, they default to "{registry_param_value}_checkpoint_path"
    and "{registry_param_value}_checkpoint_key" at injection time.
    """
    path: str
    key: str = ""
    arch_desc: str = ""
    path_field: Optional[str] = None  # YAML key for checkpoint path
    key_field: Optional[str] = None   # YAML key for checkpoint key


@dataclass
class ExperimentConfig:
    """A single fully-resolved experiment configuration ready for registration."""
    exp_id: str
    config: Dict[str, Any]
    batch_comment: str = ""


# =============================================================================
# ExperimentSeries
# =============================================================================

class ExperimentSeries:
    """
    Generate a batch of experiment configurations from a base config and a
    parameter grid via Cartesian product.

    Usage::

        series = ExperimentSeries(
            base_config={"batch_size": 64, "agent": "transfuser_agent"},
            parameter_grid={
                "backbone": ["ijepa", "dinov2"],
                "trainable": [True, False],
                "city": ["boston", "singapore"],
            },
            naming_fn=lambda p: (
                f"{p['backbone'][0].upper()}"
                f"{'t' if p['trainable'] else 'f'}"
                f"-{p['city'][0]}"
            ),
        )

        series.add_checkpoint_registry(
            {
                "ijepa":  CheckpointSpec("/scratch/.../ijepa.pth", "target_encoder"),
                "dinov2": CheckpointSpec("/scratch/.../dinov2.pth", "model"),
            },
            registry_param="backbone",
        )

        series.validate_prerequisites()   # check checkpoint paths exist
        configs = series.expand()         # returns List[ExperimentConfig]
        series.register_all(manager)      # bulk-create in manager
    """

    def __init__(
        self,
        base_config: Dict[str, Any],
        parameter_grid: Dict[str, List[Any]],
        naming_fn: Callable[[Dict[str, Any]], str],
    ):
        """
        Args:
            base_config: Base config dict applied to every experiment.
            parameter_grid: Maps parameter name -> list of values to sweep.
                            All combinations are generated (Cartesian product).
            naming_fn: Callable(params: dict) -> str returning the exp_id for
                       a given parameter combination.
                       Example:
                           lambda p: f"{p['backbone'][0].upper()}-{p['city'][0]}"
        """
        if not parameter_grid:
            raise ValueError("parameter_grid must not be empty")
        if naming_fn is None:
            raise ValueError("naming_fn is required")

        self.base_config = base_config
        self.parameter_grid = parameter_grid
        self.naming_fn = naming_fn

        self._checkpoint_registry: Dict[str, CheckpointSpec] = {}
        self._registry_param: Optional[str] = None
        self._expanded: Optional[List[ExperimentConfig]] = None

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def add_checkpoint_registry(
        self,
        registry: Dict[str, Union[CheckpointSpec, Dict[str, Any]]],
        registry_param: str
    ) -> "ExperimentSeries":
        """
        Register checkpoints keyed by parameter value.

        When expand() is called, each generated config will have its
        checkpoint path/key fields injected from the matching registry entry.

        Args:
            registry: Dict mapping a parameter value (e.g. "ijepa") to a
                      CheckpointSpec or a plain dict with keys:
                      path, key (optional), arch_desc (optional),
                      path_field (optional), key_field (optional).
            registry_param: Which parameter key to look up (e.g. "backbone").

        Returns:
            self, for chaining.
        """
        self._registry_param = registry_param
        for name, spec in registry.items():
            if isinstance(spec, CheckpointSpec):
                self._checkpoint_registry[name] = spec
            else:
                self._checkpoint_registry[name] = CheckpointSpec(**spec)
        self._expanded = None  # invalidate cache
        return self

    def validate_prerequisites(self) -> List[str]:
        """
        Check that all checkpoint paths in the registry exist on the filesystem.

        Supports glob patterns — at least one match is required.

        Returns:
            List of missing paths. Empty list means all prerequisites met.
        """
        missing = []
        for name, spec in self._checkpoint_registry.items():
            if not CheckpointResolver.exists(spec.path):
                missing.append(f"{name}: {spec.path}")
        if missing:
            print(f"WARNING: {len(missing)} checkpoint(s) missing:")
            for m in missing:
                print(f"  {m}")
        else:
            print(f"[OK] All {len(self._checkpoint_registry)} checkpoint(s) found")
        return missing

    def expand(self) -> List[ExperimentConfig]:
        """
        Generate all experiment configs from Cartesian product of parameter_grid.

        Returns:
            List of ExperimentConfig, one per unique parameter combination.

        Raises:
            ValueError if naming_fn produces duplicate exp_ids.
        """
        if self._expanded is not None:
            return self._expanded

        keys = list(self.parameter_grid.keys())
        value_lists = [self.parameter_grid[k] for k in keys]

        configs: List[ExperimentConfig] = []
        seen_ids: set = set()

        for combo in itertools.product(*value_lists):
            params = dict(zip(keys, combo))
            exp_id = self.naming_fn(params)

            if exp_id in seen_ids:
                raise ValueError(
                    f"naming_fn produced duplicate exp_id '{exp_id}' for params {params}"
                )
            seen_ids.add(exp_id)

            # Build config: base + params
            config: Dict[str, Any] = {
                **self.base_config,
                **params,
                "exp_id": exp_id,
            }

            # Inject checkpoint fields from registry
            if self._registry_param and self._checkpoint_registry:
                registry_key = params.get(self._registry_param)
                spec = self._checkpoint_registry.get(str(registry_key))
                if spec:
                    path_field = spec.path_field or f"{registry_key}_checkpoint_path"
                    key_field = spec.key_field or f"{registry_key}_checkpoint_key"
                    config[path_field] = spec.path
                    if spec.key:
                        config[key_field] = spec.key
                    if spec.arch_desc:
                        config.setdefault("arch_desc", spec.arch_desc)

            batch_comment = self._make_batch_comment(params)
            configs.append(ExperimentConfig(exp_id=exp_id, config=config, batch_comment=batch_comment))

        self._expanded = configs
        return configs

    def register_all(
        self,
        manager: "BaseExperimentManager",
        skip_existing: bool = True,
        dry_run: bool = False
    ) -> List[str]:
        """
        Bulk-create all experiments in a BaseExperimentManager instance.

        Args:
            manager: Target manager.
            skip_existing: If True, skip exp_ids already in manager.metadata.
            dry_run: Print what would be created without actually creating.

        Returns:
            List of exp_ids successfully created.
        """
        configs = self.expand()
        created = []

        for exp_cfg in configs:
            if skip_existing and exp_cfg.exp_id in manager.metadata:
                print(f"  [SKIP] {exp_cfg.exp_id} (already exists)")
                continue
            if dry_run:
                print(f"  [DRY RUN] Would create: {exp_cfg.exp_id}")
                created.append(exp_cfg.exp_id)
                continue

            # Extract known create_experiment kwargs from config
            cfg = dict(exp_cfg.config)
            description = cfg.pop("description", exp_cfg.batch_comment or "")
            cfg.pop("exp_id", None)

            manager.create_experiment(
                exp_id=exp_cfg.exp_id,
                description=description,
                **cfg
            )
            created.append(exp_cfg.exp_id)

        action = "Would create" if dry_run else "Created"
        print(f"{action} {len(created)}/{len(configs)} experiments")
        return created

    def to_yaml_files(
        self,
        output_dir: Union[str, Path],
        overwrite: bool = False
    ) -> List[Path]:
        """
        Write all expanded configs as YAML files without registering in a manager.

        Args:
            output_dir: Directory to write YAML files to.
            overwrite: If False, skip existing files.

        Returns:
            List of written file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        configs = self.expand()
        written = []

        for exp_cfg in configs:
            dest = output_dir / f"{exp_cfg.exp_id}.yaml"
            if dest.exists() and not overwrite:
                print(f"  [SKIP] {dest.name} (already exists)")
                continue
            with open(dest, "w") as f:
                if exp_cfg.batch_comment:
                    f.write(f"# {exp_cfg.batch_comment}\n")
                yaml.dump(exp_cfg.config, f, default_flow_style=False, sort_keys=False)
            written.append(dest)

        print(f"Wrote {len(written)}/{len(configs)} YAML files to {output_dir}")
        return written

    def summary(self) -> str:
        """Return a human-readable summary of this series."""
        configs = self.expand()
        lines = [f"ExperimentSeries: {len(configs)} experiments"]
        for k, vals in self.parameter_grid.items():
            lines.append(f"  {k}: {vals}")
        if self._checkpoint_registry:
            lines.append(f"  checkpoint_registry ({self._registry_param}): "
                         + ", ".join(self._checkpoint_registry.keys()))
        lines.append("  exp_ids:")
        for exp_cfg in configs:
            lines.append(f"    {exp_cfg.exp_id}")
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _make_batch_comment(self, params: Dict[str, Any]) -> str:
        """Generate a human-readable comment for the YAML header."""
        parts = [f"{k}={v}" for k, v in params.items()]
        return "  ".join(parts)
