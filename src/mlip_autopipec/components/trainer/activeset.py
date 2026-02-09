import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from mlip_autopipec.core.dataset import Dataset
from mlip_autopipec.utils.security import validate_safe_path

logger = logging.getLogger(__name__)


class ActiveSetSelector:
    """
    Selects a subset of structures using the MaxVol algorithm (via pace_activeset).
    """

    def __init__(self, limit: int = 1000) -> None:
        """
        Initialize the ActiveSetSelector.

        Args:
            limit: The maximum number of structures to select for the active set.
        """
        self.limit = limit

    def __repr__(self) -> str:
        return f"<ActiveSetSelector(limit={self.limit})>"

    def __str__(self) -> str:
        return f"ActiveSetSelector(limit={self.limit})"

    def _validate_executable(self, executable: str) -> str:
        """
        Validate that the executable is in a trusted directory.

        If PACE_ACTIVESET_BIN is set, we relax the check to a warning.
        """
        path = shutil.which(executable)
        if not path:
            msg = f"Executable '{executable}' not found in PATH."
            raise RuntimeError(msg)

        resolved_path = Path(path).resolve()
        path_str = str(resolved_path)

        # Explicitly deny /tmp and /var/tmp usage regardless of setting
        if path_str.startswith(("/tmp", "/var/tmp")):  # noqa: S108
            msg = f"Executable '{resolved_path}' is in an insecure temporary directory."
            raise SecurityError(msg)

        # Whitelist of trusted system directories
        trusted_dirs = [
            Path("/usr/bin"),
            Path("/usr/local/bin"),
            Path("/opt/bin"),
            Path.home() / ".local/bin",
        ]

        # Check if running in a virtual environment
        if "VIRTUAL_ENV" in os.environ:
            trusted_dirs.append(Path(os.environ["VIRTUAL_ENV"]) / "bin")

        is_trusted = any(str(resolved_path).startswith(str(d)) for d in trusted_dirs)

        # If explicitly configured via ENV, allow untrusted paths with warning
        is_explicitly_configured = "PACE_ACTIVESET_BIN" in os.environ

        if not is_trusted:
            if is_explicitly_configured:
                logger.warning(
                    f"Executable '{resolved_path}' is outside trusted system directories, "
                    "but allowed via PACE_ACTIVESET_BIN."
                )
            else:
                msg = (
                    f"Executable '{resolved_path}' is not in a trusted directory. "
                    f"Set PACE_ACTIVESET_BIN to override if necessary."
                )
                raise SecurityError(msg)

        return str(resolved_path)

    def _run_pace_activeset(self, input_path: Path, output_path: Path, limit: int) -> None:
        """
        Run the pace_activeset subprocess.
        """
        # Configurable executable path
        pace_bin = os.environ.get("PACE_ACTIVESET_BIN", "pace_activeset")
        pace_executable = self._validate_executable(pace_bin)

        cmd = [
            pace_executable,
            "--data-filename",
            str(input_path),
            "--output",
            str(output_path),
            "--max",
            str(limit),
        ]

        logger.info(f"Running active set selection: {' '.join(cmd)}")

        try:
            result = subprocess.run(  # noqa: S603
                cmd,
                check=True,
                capture_output=True,
                text=True,
                shell=False,
            )
            logger.debug(f"pace_activeset output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            msg = f"pace_activeset failed with error: {e.stderr}"
            logger.exception(msg)
            raise RuntimeError(msg) from e

    def select(self, input_path: Path, output_path: Path) -> Path:  # noqa: C901
        """
        Run pace_activeset to filter the dataset.
        Handles large files by chunking if necessary.

        Args:
            input_path: Path to the input dataset (gzipped pickle or extxyz).
            output_path: Path to save the selected dataset.

        Returns:
            Path to the output dataset.
        """
        # Security: Validate paths
        safe_input = validate_safe_path(input_path, must_exist=True)
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
        safe_output = output_path.resolve()
        validate_safe_path(safe_output)

        # Check file size (approximate heuristic)
        # If file is very large, pace_activeset might crash with OOM.
        # We assume 500MB is a safe limit for direct processing.
        # If larger, we split and merge.
        file_size_mb = safe_input.stat().st_size / (1024 * 1024)
        CHUNK_THRESHOLD_MB = 500

        if file_size_mb < CHUNK_THRESHOLD_MB:
            self._run_pace_activeset(safe_input, safe_output, self.limit)
            return safe_output

        logger.info(
            f"Input file size {file_size_mb:.1f}MB exceeds threshold {CHUNK_THRESHOLD_MB}MB. "
            "Using chunked active set selection."
        )

        # Chunked processing
        # Note: This requires reading the file to split it.
        # We use streaming read (Dataset) to avoid OOM in Python.
        # But we need to write chunks to disk in a format pace_activeset accepts (extxyz or pckl.gzip).
        # We assume extxyz is preferred for large files.

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            chunk_paths = []

            # Use Dataset to stream read
            # Note: root_dir is just to satisfy checks, we only read
            ds = Dataset(safe_input, root_dir=safe_input.parent)

            # Iterate in batches and write chunks
            batch_size = 5000 # Adjust as needed
            for i, batch in enumerate(ds.iter_batches(batch_size=batch_size)):
                chunk_file = tmp_path / f"chunk_{i}.extxyz"

                # Write batch to chunk file
                from ase.io import write
                atoms_list = [s.to_ase() for s in batch]
                write(chunk_file, atoms_list, format="extxyz")

                chunk_paths.append(chunk_file)

            # Select from each chunk
            selected_chunks = []
            for i, chunk_file in enumerate(chunk_paths):
                # Select a portion from each chunk
                # We aim to keep enough to satisfy the final limit.
                # E.g. select limit/N_chunks * factor?
                # Or just select 'limit' from each chunk to be safe, then merge.
                chunk_out = tmp_path / f"selected_{i}.extxyz"

                # We select 'limit' from each chunk to ensure we have candidates
                try:
                    self._run_pace_activeset(chunk_file, chunk_out, self.limit)
                    if chunk_out.exists():
                        selected_chunks.append(chunk_out)
                except Exception:
                    logger.warning(f"Failed to select from chunk {i}, skipping.")

            # Merge selected chunks
            merged_path = tmp_path / "merged_candidates.extxyz"
            # Concatenate files
            with merged_path.open("wb") as outfile:
                for fpath in selected_chunks:
                    with fpath.open("rb") as infile:
                        shutil.copyfileobj(infile, outfile)

            # Final selection on merged set
            self._run_pace_activeset(merged_path, safe_output, self.limit)

        if not safe_output.exists():
             logger.warning(f"pace_activeset finished but {safe_output} was not created.")

        return safe_output


class SecurityError(Exception):
    pass
