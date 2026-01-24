"""
Module for parsing Pacemaker training logs.
"""

import logging
import math
import re
from pathlib import Path

from mlip_autopipec.config.schemas.training import TrainingMetrics

logger = logging.getLogger(__name__)


class LogParser:
    """Parses Pacemaker log files."""

    def parse_file(self, log_path: Path) -> TrainingMetrics | None:
        """
        Parses the log file to extract the latest metrics.

        Searches for RMSE energy and force values in the log content.
        Checks for divergence (NaN/Inf).

        Args:
            log_path: Path to the log file.

        Returns:
            TrainingMetrics object or None if parsing fails.

        Raises:
            ValueError: If training diverged (NaN/Inf).
        """
        if not log_path.exists():
            logger.warning(f"Log file {log_path} does not exist.")
            return None

        try:
            content = log_path.read_text()
        except Exception as e:
            logger.error(f"Failed to read log file {log_path}: {e}")
            return None

        if not content.strip():
            return None

        # Extract Epochs
        epochs = re.findall(r"Epoch\s+(\d+)", content)
        last_epoch = int(epochs[-1]) if epochs else 0

        # Extract RMSEs (find all and take the last one)
        rmse_e_matches = re.findall(r"RMSE\s*\(energy\)\s*:\s*([^\s]+)", content)
        rmse_f_matches = re.findall(r"RMSE\s*\(forces\)\s*:\s*([^\s]+)", content)

        if not rmse_e_matches or not rmse_f_matches:
            return None

        last_rmse_e_str = rmse_e_matches[-1]
        last_rmse_f_str = rmse_f_matches[-1]

        # Check for NaN (case insensitive)
        if "nan" in last_rmse_e_str.lower() or "nan" in last_rmse_f_str.lower():
            logger.error("Training diverged: NaN detected in metrics.")
            raise ValueError("Training diverged (NaN in metrics)")

        try:
            rmse_e = float(last_rmse_e_str)
            rmse_f = float(last_rmse_f_str)
        except ValueError:
            logger.warning(
                f"Could not convert RMSE to float: E={last_rmse_e_str}, F={last_rmse_f_str}"
            )
            return None

        if not math.isfinite(rmse_e) or not math.isfinite(rmse_f):
            logger.error("Training diverged: Infinite metrics.")
            raise ValueError("Training diverged (Infinite metrics)")

        return TrainingMetrics(epoch=last_epoch, rmse_energy=rmse_e, rmse_force=rmse_f)
