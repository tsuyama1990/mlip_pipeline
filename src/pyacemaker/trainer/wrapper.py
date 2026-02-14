"""Pacemaker Wrapper."""

import subprocess
from pathlib import Path
from typing import Any


class PacemakerWrapper:
    """Wrapper for Pacemaker CLI commands."""

    def _validate_paths(self, dataset_path: Path, output_dir: Path) -> None:
        """Validate input and output paths."""
        if not dataset_path.exists():
            msg = f"Dataset path does not exist: {dataset_path}"
            raise FileNotFoundError(msg)

        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

    def _sanitize_arg(self, key: str, value: Any) -> list[str]:
        """Sanitize and format a single argument."""
        # Basic sanitization: keys should be simple strings
        if not key.replace("_", "").isalnum():
            msg = f"Invalid parameter key: {key}"
            raise ValueError(msg)

        cli_arg = f"--{key.replace('_', '-')}"

        if isinstance(value, bool):
            return [cli_arg] if value else []

        if isinstance(value, tuple | list):
            args = [cli_arg]
            for item in value:
                item_str = str(item)
                # Strict check for list items (mostly numeric expected)
                if not item_str.replace(".", "").isdigit() and not item_str.replace(
                    "-", ""
                ).isalnum():
                    msg = f"Invalid list item value: {item}"
                    raise ValueError(msg)
                args.append(item_str)
            return args

        # Sanitize scalar value
        val_str = str(value)
        # Check for dangerous shell characters
        dangerous_chars = [";", "&", "|", "$", "`", "\n", "\r"]
        if any(char in val_str for char in dangerous_chars):
            msg = f"Potential command injection in parameter value: {val_str}"
            raise ValueError(msg)
        return [cli_arg, val_str]

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
        self._validate_paths(dataset_path, output_dir)

        cmd = ["pace_train"]
        cmd.extend(["--dataset", str(dataset_path)])
        cmd.extend(["--output-dir", str(output_dir)])

        # Map params to CLI arguments
        for key, value in params.items():
            cmd.extend(self._sanitize_arg(key, value))

        if initial_potential:
            if not initial_potential.exists():
                msg = f"Initial potential path does not exist: {initial_potential}"
                raise FileNotFoundError(msg)
            cmd.extend(["--initial-potential", str(initial_potential)])

        # Capture output for logging/debugging
        # We rely on subprocess raising CalledProcessError on failure (check=True)
        # S603 ignored because we are constructing the list securely
        subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603

        # Return the path to the generated potential
        return output_dir / "output_potential.yace"

    def select_active_set(
        self, candidates_path: Path, num_select: int, output_path: Path
    ) -> Path:
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
