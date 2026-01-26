#!/usr/bin/env python3
"""
Results Harvesting Framework for ExpFlow

Provides infrastructure for harvesting training logs, evaluation results,
and generating analysis reports.

Features:
- TensorBoard log parsing
- Evaluation result extraction
- Training curve visualization
- Comparison plots across experiments
- CSV export for analysis
"""

import json
import os
import re
import glob
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# =============================================================================
# Result Data Classes
# =============================================================================

@dataclass
class TrainingMetrics:
    """Training metrics extracted from logs"""
    exp_id: str
    train_loss_last: Optional[float] = None
    train_loss_min: Optional[float] = None
    val_loss_last: Optional[float] = None
    val_loss_min: Optional[float] = None
    epochs_completed: Optional[int] = None
    final_lr: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    plot_path: Optional[str] = None


@dataclass
class EvaluationMetrics:
    """Evaluation metrics extracted from results"""
    exp_id: str
    eval_split: str
    score: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    csv_path: Optional[str] = None


# =============================================================================
# Base Results Harvester
# =============================================================================

class BaseResultsHarvester(ABC):
    """
    Base class for harvesting experiment results.

    Provides infrastructure for:
    1. Parsing training logs (TensorBoard, text logs)
    2. Extracting evaluation results
    3. Generating visualizations
    4. Exporting to CSV/JSON
    """

    def __init__(self, experiments_dir: Path, results_dir: Optional[Path] = None):
        """
        Initialize results harvester

        Args:
            experiments_dir: Root experiments directory
            results_dir: Results output directory (default: experiments_dir/results)
        """
        self.experiments_dir = Path(experiments_dir)
        self.results_dir = Path(results_dir) if results_dir else self.experiments_dir / "results"

        # Create results subdirectories
        self.plots_dir = self.results_dir / "plots"
        self.csvs_dir = self.results_dir / "csvs"
        self.analysis_dir = self.results_dir / "analysis"

        for directory in [self.results_dir, self.plots_dir, self.csvs_dir, self.analysis_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # TensorBoard Log Parsing
    # -------------------------------------------------------------------------

    def extract_tensorboard_scalars(self, event_dir: Path) -> Dict[str, Dict]:
        """
        Extract all scalar metrics from TensorBoard event files.

        Args:
            event_dir: Directory containing TensorBoard events

        Returns:
            Dictionary of metrics with values, steps, statistics
        """
        if not TENSORBOARD_AVAILABLE:
            print("Warning: TensorBoard not installed. Run: pip install tensorboard")
            return {}

        try:
            ea = EventAccumulator(str(event_dir))
            ea.Reload()

            scalars = {}
            for tag in ea.Tags().get('scalars', []):
                events = ea.Scalars(tag)
                if events:
                    scalars[tag] = {
                        'values': [e.value for e in events],
                        'steps': [e.step for e in events],
                        'last_value': events[-1].value,
                        'last_step': events[-1].step,
                        'min_value': min(e.value for e in events),
                        'max_value': max(e.value for e in events),
                        'num_steps': len(events)
                    }
            return scalars
        except Exception as e:
            print(f"Error loading TensorBoard logs from {event_dir}: {e}")
            return {}

    def find_tensorboard_logs(self, exp_id: str) -> List[Path]:
        """
        Find TensorBoard event files for an experiment.

        Args:
            exp_id: Experiment ID

        Returns:
            List of paths to event files
        """
        # Search patterns
        patterns = [
            self.experiments_dir / "training" / f"*{exp_id}*" / "**" / "events.out.tfevents*",
            self.experiments_dir / f"*{exp_id}*" / "**" / "events.out.tfevents*",
        ]

        event_files = []
        for pattern in patterns:
            event_files.extend(glob.glob(str(pattern), recursive=True))

        return [Path(f) for f in event_files]

    # -------------------------------------------------------------------------
    # Evaluation Result Parsing
    # -------------------------------------------------------------------------

    def find_evaluation_results(self, exp_id: str) -> List[Path]:
        """
        Find evaluation result files for an experiment.

        Args:
            exp_id: Experiment ID

        Returns:
            List of paths to result files (CSV, JSON, etc.)
        """
        # Search patterns
        patterns = [
            self.experiments_dir / "evaluations" / f"*{exp_id}*" / "**" / "*.csv",
            self.experiments_dir / "evaluations" / f"*{exp_id}*" / "**" / "*.json",
            self.experiments_dir / f"eval_*{exp_id}*" / "**" / "*.csv",
        ]

        result_files = []
        for pattern in patterns:
            result_files.extend(glob.glob(str(pattern), recursive=True))

        return [Path(f) for f in result_files]

    @abstractmethod
    def parse_evaluation_results(self, result_file: Path) -> EvaluationMetrics:
        """
        Parse evaluation results from a file.

        Args:
            result_file: Path to result file

        Returns:
            EvaluationMetrics object
        """
        pass

    # -------------------------------------------------------------------------
    # Training Metrics Extraction
    # -------------------------------------------------------------------------

    def harvest_training_metrics(self, exp_id: str) -> Optional[TrainingMetrics]:
        """
        Harvest training metrics for an experiment.

        Args:
            exp_id: Experiment ID

        Returns:
            TrainingMetrics object or None if not found
        """
        # Find TensorBoard logs
        event_files = self.find_tensorboard_logs(exp_id)
        if not event_files:
            return None

        # Use the first (or most recent) event file
        event_dir = event_files[0].parent
        scalars = self.extract_tensorboard_scalars(event_dir)

        if not scalars:
            return None

        # Extract key metrics
        metrics = TrainingMetrics(exp_id=exp_id)

        # Training loss
        for key in ['train_loss', 'train/loss', 'loss']:
            if key in scalars:
                metrics.train_loss_last = scalars[key]['last_value']
                metrics.train_loss_min = scalars[key]['min_value']
                break

        # Validation loss
        for key in ['val_loss', 'val/loss', 'validation_loss']:
            if key in scalars:
                metrics.val_loss_last = scalars[key]['last_value']
                metrics.val_loss_min = scalars[key]['min_value']
                break

        # Epochs
        for key in ['epoch', 'epochs']:
            if key in scalars:
                metrics.epochs_completed = int(scalars[key]['last_value'])
                break

        # Learning rate
        for key in ['lr', 'learning_rate', 'train/lr']:
            if key in scalars:
                metrics.final_lr = scalars[key]['last_value']
                break

        # Store all metrics
        metrics.metrics = scalars

        return metrics

    # -------------------------------------------------------------------------
    # Evaluation Metrics Extraction
    # -------------------------------------------------------------------------

    def harvest_evaluation_metrics(self, exp_id: str) -> List[EvaluationMetrics]:
        """
        Harvest evaluation metrics for an experiment.

        Args:
            exp_id: Experiment ID

        Returns:
            List of EvaluationMetrics objects
        """
        result_files = self.find_evaluation_results(exp_id)

        eval_metrics = []
        for result_file in result_files:
            try:
                metrics = self.parse_evaluation_results(result_file)
                if metrics:
                    eval_metrics.append(metrics)
            except Exception as e:
                print(f"Warning: Failed to parse {result_file}: {e}")

        return eval_metrics

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------

    def plot_training_curves(
        self,
        exp_id: str,
        metrics: TrainingMetrics,
        output_path: Optional[Path] = None
    ) -> Optional[Path]:
        """
        Generate training curve plots.

        Args:
            exp_id: Experiment ID
            metrics: TrainingMetrics object
            output_path: Output path for plot (default: auto-generated)

        Returns:
            Path to generated plot or None
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: Matplotlib not installed. Skipping plots.")
            return None

        if not metrics.metrics:
            return None

        scalars = metrics.metrics

        # Find loss keys
        train_key = None
        val_key = None

        for key in scalars:
            if 'train' in key.lower() and 'loss' in key.lower():
                train_key = key
            elif 'val' in key.lower() and 'loss' in key.lower():
                val_key = key

        if not train_key and not val_key:
            return None

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        if train_key:
            steps = scalars[train_key]['steps']
            values = scalars[train_key]['values']
            ax.plot(steps, values, label='Train Loss', alpha=0.8, linewidth=1.5)

        if val_key:
            steps = scalars[val_key]['steps']
            values = scalars[val_key]['values']
            ax.plot(steps, values, label='Val Loss', alpha=0.8, linewidth=1.5)

        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title(f'Training Curves: {exp_id}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Save plot
        if output_path is None:
            output_path = self.plots_dir / f"{exp_id}_training.png"

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        return output_path

    # -------------------------------------------------------------------------
    # Export Functions
    # -------------------------------------------------------------------------

    def export_to_csv(
        self,
        training_results: List[TrainingMetrics],
        eval_results: List[EvaluationMetrics],
        output_file: Optional[Path] = None
    ) -> Path:
        """
        Export results to CSV.

        Args:
            training_results: List of TrainingMetrics
            eval_results: List of EvaluationMetrics
            output_file: Output CSV path (default: auto-generated)

        Returns:
            Path to generated CSV
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.csvs_dir / f"results_{timestamp}.csv"

        # Convert to DataFrames
        train_rows = []
        for tm in training_results:
            row = {
                'exp_id': tm.exp_id,
                'train_loss_last': tm.train_loss_last,
                'train_loss_min': tm.train_loss_min,
                'val_loss_last': tm.val_loss_last,
                'val_loss_min': tm.val_loss_min,
                'epochs': tm.epochs_completed,
                'final_lr': tm.final_lr,
            }
            train_rows.append(row)

        eval_rows = []
        for em in eval_results:
            row = {
                'exp_id': em.exp_id,
                'eval_split': em.eval_split,
                'score': em.score,
                **{f'eval_{k}': v for k, v in em.metrics.items()}
            }
            eval_rows.append(row)

        # Merge
        train_df = pd.DataFrame(train_rows)
        eval_df = pd.DataFrame(eval_rows)

        if not train_df.empty and not eval_df.empty:
            merged_df = train_df.merge(eval_df, on='exp_id', how='outer')
        elif not train_df.empty:
            merged_df = train_df
        else:
            merged_df = eval_df

        # Save
        merged_df.to_csv(output_file, index=False)
        return output_file

    def export_to_json(
        self,
        training_results: List[TrainingMetrics],
        eval_results: List[EvaluationMetrics],
        output_file: Optional[Path] = None
    ) -> Path:
        """
        Export results to JSON.

        Args:
            training_results: List of TrainingMetrics
            eval_results: List of EvaluationMetrics
            output_file: Output JSON path (default: auto-generated)

        Returns:
            Path to generated JSON
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.csvs_dir / f"results_{timestamp}.json"

        results = {
            'timestamp': datetime.now().isoformat(),
            'training': [
                {
                    'exp_id': tm.exp_id,
                    'train_loss_last': tm.train_loss_last,
                    'train_loss_min': tm.train_loss_min,
                    'val_loss_last': tm.val_loss_last,
                    'val_loss_min': tm.val_loss_min,
                    'epochs': tm.epochs_completed,
                    'final_lr': tm.final_lr,
                }
                for tm in training_results
            ],
            'evaluation': [
                {
                    'exp_id': em.exp_id,
                    'eval_split': em.eval_split,
                    'score': em.score,
                    'metrics': em.metrics,
                }
                for em in eval_results
            ]
        }

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        return output_file

    # -------------------------------------------------------------------------
    # High-Level API
    # -------------------------------------------------------------------------

    def harvest_experiment(
        self,
        exp_id: str,
        generate_plots: bool = True
    ) -> Tuple[Optional[TrainingMetrics], List[EvaluationMetrics]]:
        """
        Harvest all results for an experiment.

        Args:
            exp_id: Experiment ID
            generate_plots: Whether to generate plots

        Returns:
            Tuple of (TrainingMetrics, List[EvaluationMetrics])
        """
        print(f"Harvesting results for {exp_id}...")

        # Harvest training metrics
        training_metrics = self.harvest_training_metrics(exp_id)
        if training_metrics:
            print(f"  ✓ Training: loss={training_metrics.val_loss_min:.4f}" if training_metrics.val_loss_min else "  ✓ Training metrics found")

            # Generate plots
            if generate_plots:
                plot_path = self.plot_training_curves(exp_id, training_metrics)
                if plot_path:
                    training_metrics.plot_path = str(plot_path)
                    print(f"  ✓ Plot: {plot_path}")
        else:
            print(f"  ✗ No training metrics found")

        # Harvest evaluation metrics
        eval_metrics = self.harvest_evaluation_metrics(exp_id)
        if eval_metrics:
            for em in eval_metrics:
                score_str = f"{em.score:.4f}" if em.score else "N/A"
                print(f"  ✓ Evaluation ({em.eval_split}): score={score_str}")
        else:
            print(f"  ✗ No evaluation metrics found")

        return training_metrics, eval_metrics

    def harvest_all_experiments(
        self,
        exp_ids: List[str],
        generate_plots: bool = True
    ) -> Tuple[List[TrainingMetrics], List[EvaluationMetrics]]:
        """
        Harvest results for multiple experiments.

        Args:
            exp_ids: List of experiment IDs
            generate_plots: Whether to generate plots

        Returns:
            Tuple of (List[TrainingMetrics], List[EvaluationMetrics])
        """
        all_training = []
        all_eval = []

        for exp_id in exp_ids:
            training, evaluation = self.harvest_experiment(exp_id, generate_plots)

            if training:
                all_training.append(training)

            all_eval.extend(evaluation)

        return all_training, all_eval
