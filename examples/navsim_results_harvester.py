#!/usr/bin/env python3
"""
NAVSIM Results Harvester - Built on ExpFlow

Harvests training logs and PDM Score evaluation results for NAVSIM experiments.

Usage:
    # Harvest single experiment
    python navsim_results_harvester.py harvest exp_b15

    # Harvest all experiments
    python navsim_results_harvester.py harvest-all

    # Generate comparison plots
    python navsim_results_harvester.py compare --backbone ijepa

    # Export to CSV
    python navsim_results_harvester.py export results.csv
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from expflow import BaseResultsHarvester, EvaluationMetrics, load_project_config


class NavsimResultsHarvester(BaseResultsHarvester):
    """
    Results harvester for NAVSIM experiments.

    Handles:
    - PyTorch Lightning TensorBoard logs
    - PDM Score evaluation results (CSV format)
    - Multi-stage evaluation (one-stage, two-stage)
    """

    def __init__(self):
        """Initialize NAVSIM results harvester"""

        # Load HPC config
        try:
            hpc_config = load_project_config()
        except FileNotFoundError:
            print("Error: Not in a project directory.")
            print("Run 'expflow init' first")
            sys.exit(1)

        experiments_dir = Path(hpc_config.experiments_dir)
        super().__init__(experiments_dir)

    def parse_evaluation_results(self, result_file: Path) -> Optional[EvaluationMetrics]:
        """
        Parse NAVSIM PDM Score evaluation results.

        Args:
            result_file: Path to evaluation CSV file

        Returns:
            EvaluationMetrics object
        """
        if result_file.suffix != '.csv':
            return None

        try:
            df = pd.read_csv(result_file)

            # Extract experiment ID from path
            exp_id = self._extract_exp_id_from_path(result_file)

            # Determine eval split
            eval_split = "unknown"
            parent_name = result_file.parent.name
            if "navtest" in parent_name.lower():
                eval_split = "navtest"
            elif "navhard" in parent_name.lower():
                eval_split = "navhard"
            elif "navmini" in parent_name.lower():
                eval_split = "navmini"

            # PDM Score metric columns
            metric_cols = [
                'no_at_fault_collisions',
                'drivable_area_compliance',
                'driving_direction_compliance',
                'ego_progress',
                'time_to_collision_within_bound',
                'comfort',
                'score'  # Overall PDM score
            ]

            metrics = {}
            for col in metric_cols:
                if col in df.columns:
                    metrics[col] = float(df[col].mean())

            # Overall score
            score = metrics.get('score', None)

            return EvaluationMetrics(
                exp_id=exp_id,
                eval_split=eval_split,
                score=score,
                metrics=metrics,
                csv_path=str(result_file)
            )

        except Exception as e:
            print(f"Warning: Failed to parse {result_file}: {e}")
            return None

    def _extract_exp_id_from_path(self, file_path: Path) -> str:
        """Extract experiment ID from file path"""
        # Look for exp_XXX pattern in path
        for part in file_path.parts:
            match = re.search(r'(exp_[a-z]\d+)', part, re.IGNORECASE)
            if match:
                return match.group(1)

        # Fallback to parent directory name
        return file_path.parent.name

    def find_all_experiments(self) -> List[str]:
        """Find all experiment IDs in the experiments directory"""
        exp_ids = set()

        # Search training directories
        training_dir = self.experiments_dir / "training"
        if training_dir.exists():
            for item in training_dir.iterdir():
                if item.is_dir():
                    match = re.search(r'(exp_[a-z]\d+)', item.name, re.IGNORECASE)
                    if match:
                        exp_ids.add(match.group(1))

        # Search evaluation directories
        eval_dir = self.experiments_dir / "evaluations"
        if eval_dir.exists():
            for item in eval_dir.iterdir():
                if item.is_dir():
                    match = re.search(r'(exp_[a-z]\d+)', item.name, re.IGNORECASE)
                    if match:
                        exp_ids.add(match.group(1))

        return sorted(list(exp_ids))

    def generate_comparison_plots(
        self,
        results_csv: Path,
        filter_backbone: Optional[str] = None,
        filter_agent: Optional[str] = None
    ):
        """
        Generate comparison plots from results CSV.

        Args:
            results_csv: Path to results CSV
            filter_backbone: Filter by backbone type
            filter_agent: Filter by agent type
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: Matplotlib not available. Skipping plots.")
            return

        df = pd.read_csv(results_csv)

        # Apply filters
        if filter_backbone:
            df = df[df['backbone'] == filter_backbone]
        if filter_agent:
            df = df[df['agent'] == filter_agent]

        if df.empty:
            print("No data to plot after filtering")
            return

        # Plot 1: Validation Loss Comparison
        if 'val_loss_min' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            df_sorted = df.sort_values('exp_id')

            ax.bar(df_sorted['exp_id'], df_sorted['val_loss_min'], alpha=0.7)
            ax.set_xlabel('Experiment')
            ax.set_ylabel('Min Validation Loss')
            ax.set_title('Validation Loss Comparison')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            plot_path = self.plots_dir / f"val_loss_comparison{'_' + filter_backbone if filter_backbone else ''}.png"
            plt.savefig(plot_path, dpi=150)
            plt.close(fig)
            print(f"✓ Saved: {plot_path}")

        # Plot 2: PDM Score Comparison
        if 'score' in df.columns:
            valid_scores = df[df['score'].notna()]
            if not valid_scores.empty:
                fig, ax = plt.subplots(figsize=(12, 6))
                valid_sorted = valid_scores.sort_values('exp_id')

                ax.bar(valid_sorted['exp_id'], valid_sorted['score'], alpha=0.7, color='tab:green')
                ax.set_xlabel('Experiment')
                ax.set_ylabel('PDM Score')
                ax.set_title('PDM Score Comparison')
                ax.set_ylim(0, 1)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()

                plot_path = self.plots_dir / f"pdm_score_comparison{'_' + filter_backbone if filter_backbone else ''}.png"
                plt.savefig(plot_path, dpi=150)
                plt.close(fig)
                print(f"✓ Saved: {plot_path}")

        # Plot 3: Loss vs Score scatter
        if 'val_loss_min' in df.columns and 'score' in df.columns:
            valid_both = df[df['score'].notna() & df['val_loss_min'].notna()]
            if not valid_both.empty:
                fig, ax = plt.subplots(figsize=(10, 6))

                ax.scatter(valid_both['val_loss_min'], valid_both['score'], alpha=0.6, s=100)

                # Annotate points
                for _, row in valid_both.iterrows():
                    ax.annotate(row['exp_id'], (row['val_loss_min'], row['score']),
                               fontsize=8, alpha=0.7)

                ax.set_xlabel('Min Validation Loss')
                ax.set_ylabel('PDM Score')
                ax.set_title('Validation Loss vs PDM Score')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()

                plot_path = self.plots_dir / f"loss_vs_score{'_' + filter_backbone if filter_backbone else ''}.png"
                plt.savefig(plot_path, dpi=150)
                plt.close(fig)
                print(f"✓ Saved: {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="NAVSIM Results Harvester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Harvest single experiment
  python navsim_results_harvester.py harvest exp_b15

  # Harvest all experiments
  python navsim_results_harvester.py harvest-all

  # Generate comparison plots
  python navsim_results_harvester.py compare --csv results.csv --backbone ijepa

  # Export all results
  python navsim_results_harvester.py export all_results.csv
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # harvest
    harvest_parser = subparsers.add_parser("harvest", help="Harvest single experiment")
    harvest_parser.add_argument("exp_id", help="Experiment ID")
    harvest_parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")

    # harvest-all
    harvest_all_parser = subparsers.add_parser("harvest-all", help="Harvest all experiments")
    harvest_all_parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")

    # compare
    compare_parser = subparsers.add_parser("compare", help="Generate comparison plots")
    compare_parser.add_argument("--csv", required=True, help="Results CSV file")
    compare_parser.add_argument("--backbone", help="Filter by backbone")
    compare_parser.add_argument("--agent", help="Filter by agent")

    # export
    export_parser = subparsers.add_parser("export", help="Export results to CSV")
    export_parser.add_argument("output", help="Output CSV file")
    export_parser.add_argument("--json", action="store_true", help="Also export as JSON")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    harvester = NavsimResultsHarvester()

    if args.command == "harvest":
        training, evaluation = harvester.harvest_experiment(
            args.exp_id,
            generate_plots=not args.no_plots
        )

        if training or evaluation:
            print(f"\n✓ Harvested {args.exp_id}")
            if training:
                print(f"  Training metrics saved to: {harvester.results_dir}")
            if evaluation:
                print(f"  Evaluation metrics: {len(evaluation)} results")
        else:
            print(f"\n✗ No results found for {args.exp_id}")

    elif args.command == "harvest-all":
        exp_ids = harvester.find_all_experiments()
        print(f"Found {len(exp_ids)} experiments")
        print("")

        training_results, eval_results = harvester.harvest_all_experiments(
            exp_ids,
            generate_plots=not args.no_plots
        )

        print(f"\n{'='*60}")
        print(f"Harvested {len(training_results)} training results")
        print(f"Harvested {len(eval_results)} evaluation results")
        print(f"{'='*60}")

        # Auto-export
        if training_results or eval_results:
            csv_path = harvester.export_to_csv(training_results, eval_results)
            print(f"\n✓ Exported to: {csv_path}")

            if args.json:
                json_path = harvester.export_to_json(training_results, eval_results)
                print(f"✓ Exported to: {json_path}")

    elif args.command == "compare":
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"Error: CSV file not found: {csv_path}")
            sys.exit(1)

        harvester.generate_comparison_plots(
            csv_path,
            filter_backbone=args.backbone,
            filter_agent=args.agent
        )

    elif args.command == "export":
        exp_ids = harvester.find_all_experiments()
        training_results, eval_results = harvester.harvest_all_experiments(
            exp_ids,
            generate_plots=False
        )

        output_path = Path(args.output)
        csv_path = harvester.export_to_csv(training_results, eval_results, output_path)
        print(f"✓ Exported to: {csv_path}")

        if args.json:
            json_path = output_path.with_suffix('.json')
            json_path = harvester.export_to_json(training_results, eval_results, json_path)
            print(f"✓ Exported to: {json_path}")


if __name__ == "__main__":
    main()
