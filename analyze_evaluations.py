#!/usr/bin/env python3
"""
Automated Evaluation Analysis Script
Extracts results, configs, and metadata from evaluation directories
"""

import os
import json
import yaml
import csv
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import pandas as pd

class EvaluationAnalyzer:
    def __init__(self, eval_root="evaluations"):
        self.eval_root = Path(eval_root)
        self.results = []

    def parse_experiment_name(self, dirname):
        """Extract metadata from directory name"""
        # Pattern: eval_{split}_exp_{id}_{agent}_{backbone}_{data_pct}_{timestamp}
        pattern = r"eval_(?P<split>\w+)_exp_(?P<exp_id>[ab]\d+)_(?P<agent>[^_]+(?:_agent|_velocity)?)(?:_(?P<backbone>\w+))?(?:_refactor)?_(?P<data_pct>\d+)pct_(?P<timestamp>\d{8}_\d{6})"

        match = re.match(pattern, dirname)
        if match:
            return match.groupdict()
        return {}

    def find_pdm_score(self, eval_dir):
        """Find and parse PDM score from CSV or log files"""
        # Look for CSV files with scores
        csv_files = list(eval_dir.glob("*.csv"))
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                if 'pdm_score' in df.columns:
                    return float(df['pdm_score'].iloc[-1])
            except Exception as e:
                pass

        # Look in log files
        log_file = eval_dir / "log.txt"
        if log_file.exists():
            try:
                content = log_file.read_text()
                # Look for PDM score patterns
                patterns = [
                    r"pdm[_\s]score[:\s]+([0-9.]+)",
                    r"PDMS[:\s]+([0-9.]+)",
                    r"final[_\s]score[:\s]+([0-9.]+)",
                ]
                for pattern in patterns:
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        return float(match.group(1))
            except Exception as e:
                pass

        return None

    def extract_config(self, eval_dir):
        """Extract configuration from hydra config"""
        config_file = eval_dir / "code" / "hydra" / "config.yaml"
        overrides_file = eval_dir / "code" / "hydra" / "overrides.yaml"

        config = {}

        if config_file.exists():
            try:
                with open(config_file) as f:
                    config = yaml.safe_load(f)
            except Exception as e:
                pass

        overrides = []
        if overrides_file.exists():
            try:
                with open(overrides_file) as f:
                    overrides = yaml.safe_load(f) or []
            except Exception as e:
                pass

        return {
            "full_config": config,
            "overrides": overrides,
            "agent_type": config.get("agent", {}).get("_target_", "unknown"),
            "learning_rate": config.get("agent", {}).get("learning_rate"),
            "batch_size": config.get("datamodule", {}).get("batch_size"),
            "max_epochs": config.get("trainer", {}).get("max_epochs"),
        }

    def analyze_experiment(self, eval_dir):
        """Analyze a single experiment directory"""
        dirname = eval_dir.name
        metadata = self.parse_experiment_name(dirname)

        if not metadata:
            # Try simpler pattern for cv/ego_mlp
            if "constant_velocity" in dirname:
                metadata = {"exp_id": "a1", "agent": "constant_velocity", "backbone": "none", "split": "navtest"}
            elif "ego_status_mlp" in dirname:
                metadata = {"exp_id": "a2", "agent": "ego_mlp", "backbone": "none", "split": "navtest"}

        pdm_score = self.find_pdm_score(eval_dir)
        config = self.extract_config(eval_dir)

        result = {
            "exp_id": metadata.get("exp_id", "unknown"),
            "agent": metadata.get("agent", "unknown"),
            "backbone": metadata.get("backbone", "none"),
            "data_pct": metadata.get("data_pct", "100"),
            "split": metadata.get("split", "navtest"),
            "timestamp": metadata.get("timestamp", "unknown"),
            "pdm_score": pdm_score,
            "directory": str(eval_dir.relative_to(self.eval_root)),
            **config
        }

        return result

    def analyze_all(self):
        """Analyze all evaluation directories"""
        for eval_dir in self.eval_root.rglob("eval_*"):
            if eval_dir.is_dir() and (eval_dir / "code").exists():
                try:
                    result = self.analyze_experiment(eval_dir)
                    self.results.append(result)
                    print(f"✓ Analyzed: {eval_dir.name} - Score: {result['pdm_score']}")
                except Exception as e:
                    print(f"✗ Failed: {eval_dir.name} - {e}")

        # Sort by experiment ID
        self.results.sort(key=lambda x: (x['exp_id'], x['timestamp']))

    def generate_report(self, output_file="evaluation_summary.csv"):
        """Generate summary report"""
        if not self.results:
            print("No results to report")
            return

        df = pd.DataFrame(self.results)

        # Reorder columns for readability
        column_order = [
            "exp_id", "agent", "backbone", "data_pct", "split",
            "pdm_score", "learning_rate", "batch_size", "max_epochs",
            "timestamp", "directory"
        ]

        # Only include columns that exist
        column_order = [col for col in column_order if col in df.columns]
        df = df[column_order]

        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"\n✓ Report saved to: {output_file}")

        # Print summary statistics
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)

        # Group by experiment series
        if 'exp_id' in df.columns:
            for exp_id, group in df.groupby('exp_id'):
                print(f"\n{exp_id.upper()}:")
                for _, row in group.iterrows():
                    agent = row.get('agent', 'unknown')
                    backbone = row.get('backbone', 'none')
                    data_pct = row.get('data_pct', '100')
                    score = row.get('pdm_score', 'N/A')
                    print(f"  {agent:30s} | {backbone:15s} | {data_pct:>3s}% | Score: {score}")

        # Best scores by category
        print("\n" + "-"*80)
        print("TOP PERFORMERS")
        print("-"*80)

        if 'pdm_score' in df.columns:
            df_scored = df.dropna(subset=['pdm_score'])
            if not df_scored.empty:
                top_overall = df_scored.nlargest(5, 'pdm_score')
                print("\nTop 5 Overall:")
                for idx, row in top_overall.iterrows():
                    print(f"  {row['pdm_score']:.4f} - {row['exp_id']} - {row['agent']} ({row.get('backbone', 'none')})")

        return df

def main():
    analyzer = EvaluationAnalyzer("evaluations")
    print("Starting evaluation analysis...")
    print("="*80)

    analyzer.analyze_all()

    print("\n" + "="*80)
    print(f"Analyzed {len(analyzer.results)} experiments")

    df = analyzer.generate_report("evaluation_summary.csv")

    # Also save detailed JSON
    with open("evaluation_detailed.json", "w") as f:
        json.dump(analyzer.results, f, indent=2, default=str)
    print(f"✓ Detailed JSON saved to: evaluation_detailed.json")

if __name__ == "__main__":
    main()
