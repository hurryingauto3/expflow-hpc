"""
Matrix experiment builder.

Generic M × D × ... sweep helper: takes a manager + a dict of named axes
and produces one experiment per Cartesian-product point. The manager is
responsible for the actual create/submit calls; this class just iterates
the grid and dispatches.

Extracted from navsim_manager.py:PretrainManager (the 3-method × 5-dataset
SSL pretraining matrix). Generalised so it works for any sweep shape:
``model × dataset``, ``learning_rate × seed``, ``agent × scenario_set``,
etc.
"""

from __future__ import annotations

import itertools
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


class MatrixExperimentBuilder:
    """
    Build, submit, and report on an N-axis Cartesian-product sweep.

    Args:
        manager: An object exposing ``create_experiment(exp_id, **cfg)``,
                 ``submit_experiment(exp_id, dry_run=...)``, and
                 ``metadata`` dict (any ``BaseExperimentManager`` subclass).
        axes:   Ordered dict-like mapping axis name -> list of values.
                Example: ``{"method": ["ijepa","mae"], "dataset": ["all","boston"]}``.
        naming: Function that produces an exp_id from a dict of axis-value
                bindings, e.g. ``lambda b: f"pt_{b['method']}_{b['dataset']}"``.
        config_for_point: Function ``(bindings) -> dict`` returning extra
                          config kwargs to pass to ``create_experiment``.
                          Default returns the bindings dict itself.
        template:  Optional template name to forward to
                   ``create_experiment(template=...)``.

    Example:
        builder = MatrixExperimentBuilder(
            manager,
            axes={"lr":[1e-3,1e-4], "seed":[1,2,3]},
            naming=lambda b: f"sweep_lr{b['lr']}_seed{b['seed']}",
        )
        builder.generate(dry_run=False)
        builder.submit(dry_run=True)
    """

    def __init__(
        self,
        manager: Any,
        axes: Dict[str, Iterable[Any]],
        naming: Callable[[Dict[str, Any]], str],
        *,
        config_for_point: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        template: Optional[str] = None,
    ) -> None:
        self.manager = manager
        self.axes = {k: list(v) for k, v in axes.items()}
        self.naming = naming
        self.config_for_point = config_for_point or (lambda b: dict(b))
        self.template = template

    # ── Iteration helpers ──────────────────────────────────────────────

    def points(self) -> List[Dict[str, Any]]:
        """List of binding dicts, one per Cartesian-product point."""
        if not self.axes:
            return []
        names = list(self.axes.keys())
        return [
            dict(zip(names, combo))
            for combo in itertools.product(*[self.axes[n] for n in names])
        ]

    def exp_ids(self) -> List[str]:
        """Stable list of exp_ids that this matrix would generate."""
        return [self.naming(p) for p in self.points()]

    # ── Actions ────────────────────────────────────────────────────────

    def generate(self, *, dry_run: bool = False) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Create one experiment per point.

        Returns ``[(exp_id, bindings), ...]`` for the points that were
        (or would be, in ``dry_run``) created.
        """
        out = []
        for bindings in self.points():
            exp_id = self.naming(bindings)
            kwargs = self.config_for_point(bindings)
            description = kwargs.pop(
                "description",
                f"matrix {exp_id} :: " + ", ".join(f"{k}={v}" for k, v in bindings.items()),
            )
            if dry_run:
                print(f"  [DRY RUN] would create {exp_id} with {kwargs}")
            else:
                self.manager.create_experiment(
                    exp_id,
                    template=self.template,
                    description=description,
                    **kwargs,
                )
            out.append((exp_id, bindings))
        return out

    def submit(self, *, dry_run: bool = False, **submit_kwargs: Any) -> List[str]:
        """
        Submit every experiment in the matrix that exists in metadata.

        ``submit_kwargs`` are forwarded to ``manager.submit_experiment``.
        Returns the list of exp_ids that were submitted (or would be).
        """
        submitted = []
        for exp_id in self.exp_ids():
            if exp_id not in getattr(self.manager, "metadata", {}):
                print(f"  [SKIP] {exp_id} not in metadata; run generate first")
                continue
            if dry_run:
                print(f"  [DRY RUN] would submit {exp_id}")
            else:
                self.manager.submit_experiment(exp_id, dry_run=False, **submit_kwargs)
            submitted.append(exp_id)
        return submitted

    def status_grid(self) -> Dict[Tuple, str]:
        """
        Return a dict keyed by axis-value tuple → status string from manager metadata.

        Useful for printing an axis × axis grid view of the sweep.
        """
        metadata = getattr(self.manager, "metadata", {})
        out: Dict[Tuple, str] = {}
        names = list(self.axes.keys())
        for bindings in self.points():
            exp_id = self.naming(bindings)
            entry = metadata.get(exp_id)
            status = entry.get("status", "—") if isinstance(entry, dict) else "—"
            key = tuple(bindings[n] for n in names)
            out[key] = status
        return out
