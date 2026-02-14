"""Pacemaker Wrapper."""

import subprocess
from pathlib import Path
from typing import Any


class PacemakerWrapper:
    """Wrapper for Pacemaker CLI commands."""

    def train(
        self,
        dataset_path: Path,
        output_dir: Path,
        params: dict[str, Any],
        initial_potential: Path | None = None,
    ) -> Path:
        """Run pace_train command.

        Args:
            dataset_path: Path to dataset (.pckl.gzip)
            output_dir: Output directory
            params: Dictionary of parameters (cutoff, order, etc.)
            initial_potential: Optional initial potential for fine-tuning

        Returns:
            Path to the trained potential file.

        """
        cmd = ["pace_train"]
        cmd.extend(["--dataset", str(dataset_path)])
        cmd.extend(["--output-dir", str(output_dir)])

        # Map params to CLI arguments
        for key, value in params.items():
            # Convert snake_case to kebab-case
            cli_arg = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    cmd.append(cli_arg)
            elif isinstance(value, tuple | list):
                # Handle tuple/list arguments
                # E.g. basis_size=(15, 5) -> --basis-size 15 5
                cmd.append(cli_arg)
                cmd.extend(map(str, value))
            else:
                cmd.extend([cli_arg, str(value)])

        if initial_potential:
            cmd.extend(["--initial-potential", str(initial_potential)])

        # Capture output for logging/debugging
        # We rely on subprocess raising CalledProcessError on failure (check=True)
        subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603

        # Return the path to the generated potential
        # Assuming standard naming convention or user knows where to look in output_dir
        # For now, return a generic path that Trainer can verify
        return output_dir / "output_potential.yace"

    def select_active_set(self, candidates_path: Path, num_select: int, output_path: Path) -> Path:
        """Run pace_activeset command.

        Args:
            candidates_path: Path to candidates dataset
            num_select: Number of structures to select
            output_path: Path to save selected structures

        Returns:
            Path to the selected dataset.

        """
        cmd = ["pace_activeset"]
        cmd.extend(["--dataset", str(candidates_path)])
        cmd.extend(["--select", str(num_select)])
        cmd.extend(["--output", str(output_path)])

        subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603

        return output_path
