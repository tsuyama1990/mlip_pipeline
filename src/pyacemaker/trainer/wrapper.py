"""Pacemaker Wrapper."""

import re
import subprocess
from pathlib import Path
from typing import Any

from pyacemaker.core.validation import validate_safe_path


class PacemakerWrapper:
    """Wrapper for Pacemaker CLI commands."""

    def _validate_paths(self, dataset_path: Path, output_dir: Path) -> None:
        """Validate input and output paths."""
        if not dataset_path.exists():
            msg = f"Dataset path does not exist: {dataset_path}"
            raise FileNotFoundError(msg)

        # Validate safety
        validate_safe_path(dataset_path)
        validate_safe_path(output_dir)

        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

    def _sanitize_arg(self, key: str, value: Any) -> list[str]:
        """Format a single argument.

        Relies on subprocess.run(shell=False) for safety of values.
        Validates keys to ensure they are simple flags.
        Also validates values against simple regex to prevent weird chars.
        """
        # Validate key (flags) to be safe
        if not key.replace("_", "").isalnum():
            msg = f"Invalid parameter key: {key}"
            raise ValueError(msg)

        cli_arg = f"--{key.replace('_', '-')}"

        if isinstance(value, bool):
            return [cli_arg] if value else []

        if isinstance(value, tuple | list):
            args = [cli_arg]
            for item in value:
                val_str = str(item)
                if not re.match(r"^[a-zA-Z0-9_.\-+,/]+$", val_str):
                     # Allow alphanumeric, underscore, dot, dash, plus, comma, slash (for paths)
                     # Stricter than shell injection because subprocess handles that, but good practice
                     msg = f"Invalid parameter value: {val_str}"
                     raise ValueError(msg)
                args.append(val_str)
            return args

        val_str = str(value)
        if not re.match(r"^[a-zA-Z0-9_.\-+,/ :]+$", val_str):
             # Allow colon and space too
             msg = f"Invalid parameter value: {val_str}"
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
            validate_safe_path(initial_potential)
            cmd.extend(["--initial-potential", str(initial_potential)])

        # Capture output for logging/debugging
        # We rely on subprocess raising CalledProcessError on failure (check=True)
        # S603 ignored because we are constructing the list securely
        subprocess.run(
            cmd, check=True, capture_output=True, text=True, shell=False
        )

        # Return the path to the generated potential
        return output_dir / "output_potential.yace"

    def train_from_input(
        self,
        input_file: Path,
        output_dir: Path,
        initial_potential: Path | None = None
    ) -> Path:
        """Run pace_train with input.yaml.

        Args:
            input_file: Path to input.yaml
            output_dir: Output directory (where potential.yace is expected to be)
            initial_potential: Optional initial potential for fine-tuning

        Returns:
            Path to the trained potential file.
        """
        validate_safe_path(input_file)
        validate_safe_path(output_dir)

        if not input_file.exists():
            msg = f"Input file does not exist: {input_file}"
            raise FileNotFoundError(msg)

        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        cmd = ["pace_train", str(input_file)]

        if initial_potential:
            validate_safe_path(initial_potential)
            if not initial_potential.exists():
                msg = f"Initial potential path does not exist: {initial_potential}"
                raise FileNotFoundError(msg)
            cmd.extend(["--initial-potential", str(initial_potential)])

        log_path = output_dir / "pace_train.log"
        with log_path.open("w") as log_file:
            # S603 ignored because we are constructing the list securely
            # Use cwd=output_dir to ensure relative paths in input.yaml (like output filenames) resolve correctly
            # Redirect output to file to avoid OOM
            subprocess.run(
                cmd,
                check=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=output_dir,
                shell=False,
            )

        # Return the path to the generated potential
        # Note: Pacemaker input.yaml usually specifies output file name.
        # We assume standard output or check where it went.
        # But commonly it produces potential.yace in CWD or specified output.
        # If input.yaml specifies output path relative to CWD, we need to know it.
        # We assume for now it is "potential.yace" in the CWD of execution?
        # subprocess.run uses default CWD unless specified.
        # If input.yaml has paths, they are relative to CWD.
        # We should probably run in output_dir or ensure input.yaml paths are absolute.
        # But input_file is passed as argument.

        return output_dir / "potential.yace"

    def select_active_set(self, candidates_path: Path, num_select: int, output_path: Path) -> Path:
        """Run pace_activeset command.

        Args:
            candidates_path: Path to candidates dataset
            num_select: Number of structures to select
            output_path: Path to save selected structures

        Returns:
            Path to the selected dataset.

        """
        # Validate inputs strictly even here
        validate_safe_path(candidates_path)
        validate_safe_path(output_path)

        if not candidates_path.exists():
            msg = f"Candidates path does not exist: {candidates_path}"
            raise FileNotFoundError(msg)

        # Ensure num_select is positive integer
        if num_select <= 0:
            msg = f"Number of structures to select must be positive: {num_select}"
            raise ValueError(msg)

        cmd = ["pace_activeset"]
        cmd.extend(["--dataset", str(candidates_path)])
        cmd.extend(["--select", str(num_select)])
        cmd.extend(["--output", str(output_path)])

        subprocess.run(
            cmd, check=True, capture_output=True, text=True, shell=False
        )

        return output_path
