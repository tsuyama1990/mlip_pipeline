import logging
import random
from collections.abc import Generator
from pathlib import Path

from ase import Atoms
from ase.io import write

from mlip_autopipec.orchestration.database import DatabaseManager

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """
    Constructs training datasets by querying the database and exporting to file.
    Supports streaming for scalability.
    """

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def fetch_data(
        self, query: str = "converged=True", limit: int | None = None
    ) -> Generator[Atoms, None, None]:
        """
        Fetches atoms from database matching query.
        Returns a generator to avoid loading all into memory.
        """
        logger.info(f"Fetching training data with query: '{query}'")

        # Use database streaming
        count = 0
        for atoms in self.db_manager.select(selection=query):
            yield atoms
            count += 1
            if limit and count >= limit:
                break

        if count == 0:
            logger.warning(f"No atoms found with query '{query}'")

    def export(
        self,
        output_path: Path,
        query: str = "converged=True",
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> None:
        """
        Exports data to extended XYZ format, splitting into train/test sets.
        Uses streaming write to handle large datasets.
        """
        train_path = output_path.parent / "train.xyz"
        test_path = output_path.parent / "test.xyz"

        # Clean up existing
        if train_path.exists():
            train_path.unlink()
        if test_path.exists():
            test_path.unlink()

        logger.info(f"Exporting dataset to {train_path} and {test_path} (Ratio: {test_ratio})")

        rng = random.Random(seed)
        count = 0
        train_count = 0
        test_count = 0

        # Open files for streaming write
        try:
            # We use ase.io.write with append=True or a specialized streamer.
            # ASE 'extxyz' supports appending.
            # However, opening and closing file for every atom is slow.
            # Better to keep handle open, but ASE write accepts filename or file object.
            # Let's try passing file objects.

            with open(train_path, "w") as f_train, open(test_path, "w") as f_test:
                for atoms in self.fetch_data(query):
                    count += 1
                    if rng.random() < test_ratio:
                        write(f_test, atoms, format="extxyz")
                        test_count += 1
                    else:
                        write(f_train, atoms, format="extxyz")
                        train_count += 1

        except Exception as e:
            logger.exception(f"Error during data export: {e}")
            raise

        if count == 0:
            raise ValueError("No training data found in database.")

        if test_count == 0 and train_count > 0:
            logger.warning("Test set is empty. Consider increasing test_ratio or data size.")

        logger.info(f"Export complete. Total: {count}, Train: {train_count}, Test: {test_count}")
