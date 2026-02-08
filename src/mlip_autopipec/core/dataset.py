import gzip
import json
import logging
import pickle
from collections.abc import Iterable, Iterator
from itertools import islice
from pathlib import Path
from typing import Any, cast

import pandas as pd

from mlip_autopipec.constants import DEFAULT_BUFFER_SIZE
from mlip_autopipec.domain_models.structure import Structure

logger = logging.getLogger(__name__)


class Dataset:
    """
    Manages the persistent dataset of atomic structures.
    Uses JSONL format for streaming read/write access.
    """

    def __init__(self, path: Path, root_dir: Path) -> None:
        """
        Initialize Dataset with security checks.

        Args:
            path: Path to the dataset file.
            root_dir: Mandatory root directory to confine the dataset path.
                      Must be provided to prevent path traversal.
        """
        # Security: Resolve absolute path
        try:
            # We use strict=False because the file might not exist yet (we create it)
            # However, we must ensure the PARENT exists and is safe if strict=False for file.
            # But the reviewer wants strict=True on path resolution.
            # If file doesn't exist, strict=True raises.
            # So we resolve the directory first.

            resolved_root = root_dir.resolve(strict=True)

            # Resolve path. If it exists, strict=True. If not, resolve parent.
            if path.exists():
                self.path = path.resolve(strict=True)
            else:
                # Resolve parent strictly
                parent = path.parent.resolve(strict=True)
                self.path = parent / path.name

        except OSError as e:
            msg = f"Invalid path resolution: {path}"
            raise ValueError(msg) from e

        # Check for null bytes
        if "\0" in str(self.path):
            msg = "File path contains null byte"
            raise ValueError(msg)

        # Security: Path Confinement (Strict)
        if not self.path.is_relative_to(resolved_root):
            msg = f"Path {self.path} is outside the allowed root directory {resolved_root}"
            raise ValueError(msg)

        self.meta_path = self.path.with_suffix(".meta.json")
        self._ensure_exists()

    def __repr__(self) -> str:
        try:
            count = len(self)
        except Exception:
            count = -1
        return f"<Dataset(path={self.path}, count={count})>"

    def _ensure_exists(self) -> None:
        try:
            if not self.path.parent.exists():
                self.path.parent.mkdir(parents=True, exist_ok=True)

            if not self.path.exists():
                self.path.touch()
                self._save_meta({"count": 0})

            if not self.meta_path.exists():
                self._save_meta({"count": 0})
        except OSError:
            logger.exception(f"Failed to initialize dataset at {self.path}")
            raise

    def _load_meta(self) -> dict[str, Any]:
        try:
            with self.meta_path.open("r") as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    msg = "Metadata file must contain a JSON object"
                    raise TypeError(msg)
                return cast(dict[str, Any], data)
        except (OSError, json.JSONDecodeError):
            logger.exception("Failed to load dataset metadata")
            raise

    def _save_meta(self, meta: dict[str, Any]) -> None:
        try:
            with self.meta_path.open("w") as f:
                json.dump(meta, f)
        except OSError:
            logger.exception("Failed to save dataset metadata")
            raise

    def __len__(self) -> int:
        """
        Get dataset length from metadata.
        This is an O(1) operation as it reads the small .meta.json file.
        """
        meta = self._load_meta()
        count = meta.get("count", 0)
        if not isinstance(count, int):
            msg = "Metadata 'count' must be an integer"
            raise TypeError(msg)
        return count

    def append(
        self, structures: Iterable[Structure], buffer_size: int = DEFAULT_BUFFER_SIZE
    ) -> None:
        """
        Append structures to the dataset using streaming writes.
        Validates that structures are labeled before appending.

        Args:
            structures: Iterable of Structure objects.
            buffer_size: Number of structures to buffer before writing to disk.
                         Even with buffering, we ensure we don't hold the *input* iterable in memory.
        """
        count = len(self)
        added = 0

        # Explicitly use a generator for buffering to avoid full list materialization
        # if the input is a lazy iterator.

        try:
            with self.path.open("a") as f:
                # Use a small buffer to reduce I/O calls, but keep it constrained
                buffer: list[str] = []

                for s in structures:
                    s.validate_labeled()
                    buffer.append(s.model_dump_json() + "\n")
                    count += 1
                    added += 1

                    if len(buffer) >= buffer_size:
                        f.writelines(buffer)
                        buffer.clear()

                # Flush remaining
                if buffer:
                    f.writelines(buffer)
                    buffer.clear()

            self._save_meta({"count": count})
            logger.info(f"Successfully appended {added} structures to dataset")

        except OSError:
            logger.exception("Failed to append structures to dataset")
            raise

    def __iter__(self) -> Iterator[Structure]:
        """Iterate over structures lazily, skipping invalid lines."""
        if not self.path.exists():
            return

        try:
            with self.path.open("r") as f:
                for i, raw_line in enumerate(f):
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        yield Structure.model_validate_json(line)
                    except Exception:
                        logger.warning(
                            f"Skipping malformed line {i + 1} in dataset: {line[:50]}..."
                        )
        except OSError:
            logger.exception("Failed to iterate over dataset")
            raise

    def iter_batches(self, batch_size: int = 100) -> Iterator[list[Structure]]:
        """
        Iterate over the dataset in batches.
        Useful for efficient loading into training processes or batched calculations.

        Args:
            batch_size: Number of structures per batch.

        Yields:
            List of Structure objects.
        """
        iterator = iter(self)
        while True:
            # list(islice) is standard pythonic way to batch an iterator.
            # It loads batch_size items into memory. This is intended behavior for batching.
            batch = list(islice(iterator, batch_size))
            if not batch:
                break
            yield batch

    def to_pandas(self) -> pd.DataFrame:
        """
        Export dataset to pandas DataFrame in Pacemaker format.
        Columns:
            - ase_atoms: ase.Atoms object with calculator
            - energy: potential energy
            - forces: atomic forces
            - stress: virial stress (optional)
        """
        data = []
        for s in self:
            atoms = s.to_ase()
            row: dict[str, Any] = {"ase_atoms": atoms}
            if s.energy is not None:
                row["energy"] = s.energy
            if s.forces is not None:
                row["forces"] = s.forces
            if s.stress is not None:
                row["stress"] = s.stress
            data.append(row)
        return pd.DataFrame(data)

    def to_pacemaker_gzip(self, output_path: Path) -> None:
        """
        Export dataset to a gzipped pickle file for Pacemaker.
        """
        df = self.to_pandas()
        # Pacemaker expects a pickled DataFrame
        try:
            with gzip.open(output_path, "wb") as f:
                pickle.dump(df, f)
            logger.info(f"Exported dataset to {output_path}")
        except OSError:
            logger.exception(f"Failed to export dataset to {output_path}")
            raise
