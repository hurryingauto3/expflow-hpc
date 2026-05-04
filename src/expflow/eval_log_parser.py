"""
Log-fallback parsing for evaluation runs.

When a CSV result file is missing or unparseable, fall back to scraping
metrics out of the SLURM ``.out`` log file with a configurable set of
regex patterns. Captures the score, the scenario count, and a status
classification (SUCCESS / FAILED / UNKNOWN).

Extracted from navsim_manager.py:_pick_latest_log + _parse_eval_log
so any project that emits eval logs in a known format can wire it up
without re-implementing the discovery + regex logic.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Pattern


class EvalLogParser:
    """
    Parse evaluation metrics from SLURM stdout logs.

    Args:
        score_patterns: Regex patterns whose first capture group is a
                        floating-point score. The first match wins.
        scenarios_patterns: Regex patterns whose first capture group is
                            an integer scenario count. First match wins.
        success_pattern: Regex that, when present in the log, marks the
                         run as SUCCESS.
        failure_pattern: Regex that marks the run as FAILED.

    Defaults are tuned to NAVSIM's PDM-Score logger but every pattern
    is overridable.
    """

    DEFAULT_SCORE_PATTERNS = [
        r"Final average score of valid results:\s*([0-9]+\.?[0-9]*)",
        r"PDMS:\s*([0-9.]+)",
        r"final[_ ]score[=:\s]+([0-9]+\.?[0-9]*)",
    ]
    DEFAULT_SCENARIOS_PATTERNS = [
        r"Number of successful scenarios:\s*(\d+)",
        r"scenarios[_ ]evaluated[=:\s]+(\d+)",
    ]
    DEFAULT_SUCCESS_PATTERN = r"Status:\s*SUCCESS"
    DEFAULT_FAILURE_PATTERN = r"Status:\s*FAILED"

    def __init__(
        self,
        score_patterns: Optional[List[str]] = None,
        scenarios_patterns: Optional[List[str]] = None,
        success_pattern: Optional[str] = None,
        failure_pattern: Optional[str] = None,
    ) -> None:
        self._score_patterns: List[Pattern] = [
            re.compile(p) for p in (score_patterns or self.DEFAULT_SCORE_PATTERNS)
        ]
        self._scenarios_patterns: List[Pattern] = [
            re.compile(p)
            for p in (scenarios_patterns or self.DEFAULT_SCENARIOS_PATTERNS)
        ]
        self._success_re = re.compile(success_pattern or self.DEFAULT_SUCCESS_PATTERN)
        self._failure_re = re.compile(failure_pattern or self.DEFAULT_FAILURE_PATTERN)

    def latest_log(self, log_dir: Path, patterns: List[str]) -> Optional[Path]:
        """
        Pick the most-recent log under ``log_dir`` matching any of ``patterns``.

        Tie-breaks on a numeric job ID embedded as ``_<digits>.out`` (so
        higher SLURM job IDs win for runs queued the same second), then
        on file mtime.
        """
        candidates: List[Path] = []
        for pat in patterns:
            candidates.extend(log_dir.glob(pat))
        if not candidates:
            return None

        def _job_id(path: Path) -> int:
            m = re.search(r"_(\d+)\.out$", path.name)
            return int(m.group(1)) if m else 0

        return max(candidates, key=lambda p: (_job_id(p), p.stat().st_mtime))

    def parse(self, log_file: Path) -> Dict[str, object]:
        """
        Read ``log_file`` and extract score / scenarios / status.

        Returns a dict with keys: ``score`` (float | None),
        ``scenarios`` (int), ``status`` (``"SUCCESS"`` | ``"FAILED"`` |
        ``"UNKNOWN"``), ``log_file`` (str).
        """
        try:
            content = log_file.read_text(errors="replace")
        except OSError:
            return {
                "score": None,
                "scenarios": 0,
                "status": "UNKNOWN",
                "log_file": str(log_file.name),
            }

        score: Optional[float] = None
        for rx in self._score_patterns:
            m = rx.search(content)
            if m:
                try:
                    score = float(m.group(1))
                    break
                except (TypeError, ValueError):
                    continue

        scenarios = 0
        for rx in self._scenarios_patterns:
            m = rx.search(content)
            if m:
                try:
                    scenarios = int(m.group(1))
                    break
                except (TypeError, ValueError):
                    continue

        if self._failure_re.search(content):
            status = "FAILED"
        elif self._success_re.search(content):
            status = "SUCCESS"
        elif score is not None and score > 0:
            status = "SUCCESS"
        else:
            status = "UNKNOWN"

        return {
            "score": score,
            "scenarios": scenarios,
            "status": status,
            "log_file": str(log_file.name),
        }
