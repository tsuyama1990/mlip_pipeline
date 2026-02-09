import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from mlip_autopipec.core.dataset import Dataset
from mlip_autopipec.utils.security import validate_safe_path

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Exception raised for security violations."""


class ActiveSetSelector:
    """
    Selects a subset of structures using the MaxVol algorithm (via pace_activeset).
    """

    def __init__(
        self,
        limit: int = 1000,
        chunk_threshold_mb: int = 500,
        batch_size: int = 5000
    ) -> None:
        """
        Initialize the ActiveSetSelector.

        Args:
            limit: The maximum number of structures to select for the active set.
            chunk_threshold_mb: File size threshold (MB) to trigger chunked processing.
            batch_size: Number of structures per chunk.
        """
        if limit <= 0:
            msg = f"Limit must be positive, got {limit}"
            raise ValueError(msg)
        self.limit = limit
        self.chunk_threshold_mb = chunk_threshold_mb
        self.batch_size = batch_size

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
        forbidden_prefixes = ("/tmp", "/var/tmp")  # noqa: S108
        if path_str.startswith(forbidden_prefixes):
            msg = f"Executable '{resolved_path}' is in an insecure temporary directory."
            raise SecurityError(msg)

        # Whitelist of trusted system directories
        trusted_dirs = [
            Path("/usr/bin"),
            Path("/usr/local/bin"),
            # Removed /opt/bin per security audit
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

    def select(self, input_path: Path, output_path: Path) -> Path:
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
        file_size_mb = safe_input.stat().st_size / (1024 * 1024)

        if file_size_mb < self.chunk_threshold_mb:
            self._run_pace_activeset(safe_input, safe_output, self.limit)
            return safe_output

        logger.info(
            f"Input file size {file_size_mb:.1f}MB exceeds threshold {self.chunk_threshold_mb}MB. "
            "Using chunked active set selection."
        )

        # Chunked processing
        self._process_chunks(safe_input, safe_output)

        if not safe_output.exists():
             logger.warning(f"pace_activeset finished but {safe_output} was not created.")

        return safe_output

    def _process_chunks(self, input_path: Path, output_path: Path) -> None:
        """Process large files by splitting into chunks, selecting, and merging."""
        from ase.io import write

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            merged_path = tmp_path / "merged_candidates.extxyz"

            # Use Dataset to stream read
            # Note: root_dir is just to satisfy checks, we only read
            ds = Dataset(input_path, root_dir=input_path.parent)

            # Process batches sequentially
            # Read batch -> Write chunk -> Select from chunk -> Append to merge -> Delete chunk
            # This minimizes disk usage and memory footprint

            # Initialize merge file
            merged_path.touch()

            for i, batch in enumerate(ds.iter_batches(batch_size=self.batch_size)):
                chunk_file = tmp_path / f"chunk_{i}.extxyz"
                chunk_out = tmp_path / f"selected_{i}.extxyz"

                try:
                    # Write batch to chunk file
                    # Batch is a list of Structures.
                    # We stream write if possible, but ase.io.write takes list or atoms.
                    # Given batch_size is small (5000), list is acceptable.
                    atoms_list = [s.to_ase() for s in batch]
                    write(chunk_file, atoms_list, format="extxyz")

                    # Select from chunk
                    self._run_pace_activeset(chunk_file, chunk_out, self.limit)

                    # Append directly to merged file
                    if chunk_out.exists():
                        with merged_path.open("ab") as outfile, chunk_out.open("rb") as infile:
                            shutil.copyfileobj(infile, outfile)

                except Exception:
                    logger.warning(f"Failed to process chunk {i}, skipping.")
                finally:
                    # Cleanup immediate chunk files
                    if chunk_file.exists():
                        chunk_file.unlink()
                    if chunk_out.exists():
                        chunk_out.unlink()

            # Final selection on merged set
            self._run_pace_activeset(merged_path, output_path, self.limit)
