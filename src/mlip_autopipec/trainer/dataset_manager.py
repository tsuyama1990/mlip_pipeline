import logging
import subprocess
from collections.abc import Iterable
from pathlib import Path

from ase import Atoms
from ase.io import write

from mlip_autopipec.domain_models.structure import Structure

logger = logging.getLogger(__name__)


class DatasetManager:
    """Manages dataset collection and active set selection using Pacemaker."""

    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def create_dataset(self, structures: Iterable[Structure], output_path: Path) -> tuple[Path, list[str], int]:
        """
        Converts structures to Pacemaker's dataset format.

        Args:
            structures: Iterable of structures to convert.
            output_path: Path to the output dataset file (.pckl.gzip).

        Returns:
            Tuple containing:
            - Path to the created dataset file.
            - List of unique chemical symbols found in the dataset.
            - Count of structures written.
        """
        # 1. Write structures to a temporary extxyz file using streaming
        temp_extxyz = self.work_dir / "temp_structures.extxyz"
        elements: set[str] = set()
        count = 0

        def structure_generator() -> Iterable[Atoms]:
            nonlocal count
            for s in structures:
                atoms = s.to_ase()
                # Update elements set
                # get_chemical_symbols returns a list of strings
                unique_syms = set(atoms.get_chemical_symbols()) # type: ignore[no-untyped-call]
                elements.update(unique_syms)
                count += 1
                yield atoms

        # Write streaming to disk to avoid OOM
        try:
            # Cast generator to Any to satisfy mypy's expectation for write() which might be strict
            write(str(temp_extxyz), structure_generator(), format="extxyz") # type: ignore[arg-type]
        except Exception as e:
            logger.exception(f"Failed to write temporary structure file: {e}")
            raise

        # 2. Call pace_collect
        # Usage: pace_collect input_file.extxyz --output dataset.pckl.gzip
        cmd = [
            "pace_collect",
            temp_extxyz.name,
            "--output",
            output_path.name,
        ]

        logger.info(f"Running pace_collect: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=self.work_dir)  # noqa: S603

        if result.returncode != 0:
            msg = f"pace_collect failed: {result.stderr}"
            logger.error(msg)
            # Try to cleanup
            temp_extxyz.unlink(missing_ok=True)
            raise RuntimeError(msg)

        if not output_path.exists():
            msg = f"pace_collect did not produce output file: {output_path}"
            logger.error(msg)
            # Try to cleanup
            temp_extxyz.unlink(missing_ok=True)
            raise FileNotFoundError(msg)

        # Clean up temp file
        temp_extxyz.unlink(missing_ok=True)

        return output_path, sorted(elements), count

    def select_active_set(self, dataset_path: Path, count: int) -> Path:
        """
        Selects an active set from the dataset.

        Args:
            dataset_path: Path to the full dataset (.pckl.gzip).
            count: Number of structures to select.

        Returns:
            Path to the new dataset containing only the active set.
        """
        output_path = self.work_dir / f"active_set_{count}.pckl.gzip"

        # Usage: pace_activeset dataset.pckl.gzip --max_size 100 --output active_set.pckl.gzip
        # Assuming dataset_path is within work_dir
        cmd = [
            "pace_activeset",
            dataset_path.name,
            "--max_size",
            str(count),
            "--output",
            output_path.name,
        ]

        logger.info(f"Running pace_activeset: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=self.work_dir)  # noqa: S603

        if result.returncode != 0:
            msg = f"pace_activeset failed: {result.stderr}"
            logger.error(msg)
            raise RuntimeError(msg)

        if not output_path.exists():
            msg = f"pace_activeset did not produce output file: {output_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        return output_path
