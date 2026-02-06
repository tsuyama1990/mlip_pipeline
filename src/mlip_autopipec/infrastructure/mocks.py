import logging
import uuid
from collections.abc import Generator
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import iread, write

from mlip_autopipec.config import TrainerConfig, ValidatorConfig
from mlip_autopipec.domain_models import Dataset, StructureMetadata, ValidationResult
from mlip_autopipec.interfaces import BaseExplorer, BaseOracle, BaseTrainer, BaseValidator

logger = logging.getLogger(__name__)

class MockExplorer(BaseExplorer):
    """
    Mock implementation of an Explorer that generates random H2O structures.
    """
    def explore(self, current_potential_path: Path, dataset: Dataset) -> Dataset:
        """
        Generates dummy candidate structures.
        """
        logger.info("MockExplorer: Generating new candidates...")

        def _generate() -> Generator[StructureMetadata, None, None]:
            for _ in range(2):
                atoms = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])
                atoms.positions += np.random.rand(3, 3) * 0.1
                yield StructureMetadata(structure=atoms, iteration=0)

        new_structures = list(_generate())
        logger.info(f"MockExplorer: Generated {len(new_structures)} structures.")
        return Dataset(structures=new_structures)

class MockOracle(BaseOracle):
    """
    Mock implementation of an Oracle that assigns random energies and forces.
    """
    def __init__(self, work_dir: Path | None = None) -> None:
        super().__init__(work_dir)
        # Ensure work_dir is set if we need to write files
        if self.work_dir:
            self.work_dir.mkdir(parents=True, exist_ok=True)

    def label(self, dataset: Dataset) -> Dataset:
        """
        Labels the dataset with random physical properties.
        Supports in-memory and file-based datasets (streaming).
        """

        def _label_structure(meta: StructureMetadata) -> StructureMetadata:
             # Simulate labeling
            meta.energy = float(np.random.uniform(-100, -10))
            n_atoms = len(meta.structure)
            meta.forces = np.random.uniform(-1, 1, size=(n_atoms, 3)).tolist()
            meta.virial = np.random.uniform(-0.1, 0.1, size=(3, 3)).tolist()
            return meta

        # Case 1: File-based dataset (Stream processing)
        if dataset.file_path and not dataset.structures:
            logger.info(f"MockOracle: Streaming and labeling from file {dataset.file_path}...")

            if not self.work_dir:
                 # Fallback if work_dir not provided, though ideally it should be.
                 output_file = Path(f"labeled_{uuid.uuid4()}.xyz")
            else:
                 output_file = self.work_dir / f"labeled_{uuid.uuid4()}.xyz"

            count = 0
            # Open output file for appending/writing
            # We use ase.io.write in a loop or collect batches.
            # Since we want to stream, we can't easily append to standard formats like xyz efficiently without keeping file open.
            # But ASE write(append=True) works.

            # Use iread to stream input
            # Note: Dataset.file_path is Path, ase.io.iread expects str or path-like
            try:
                # We need to construct StructureMetadata from Atoms read from file
                # But typically Oracle reads Atoms, labels them.
                # Here we assume the input file contains atoms.
                for atoms in iread(str(dataset.file_path)):
                    # Convert to StructureMetadata (dummy iteration)
                    meta = StructureMetadata(structure=atoms, iteration=0)
                    labeled_meta = _label_structure(meta)

                    # Write immediately (inefficient but safe for OOM)
                    # Or batch it. Let's do immediate for simplicity of "streaming" proof.
                    write(output_file, labeled_meta.structure, append=True)
                    count += 1
            except Exception as e:
                msg = f"Failed to stream/label from {dataset.file_path}"
                logger.exception(msg)
                raise RuntimeError(msg) from e

            logger.info(f"MockOracle: Labeled {count} structures -> {output_file}")
            return Dataset(file_path=output_file)

        # Case 2: In-memory dataset
        logger.info(f"MockOracle: Labeling {len(dataset.structures)} structures in memory...")
        labeled_structures: list[StructureMetadata] = []

        for meta in dataset.structures:
            if meta.structure is None:
                logger.warning("MockOracle: Encountered StructureMetadata with None structure.")
                continue
            labeled_structures.append(_label_structure(meta))

        logger.info("MockOracle: Labeling complete.")
        return Dataset(structures=labeled_structures)

class MockTrainer(BaseTrainer):
    """
    Mock implementation of a Trainer that produces a dummy potential file.
    """
    def __init__(self, config: TrainerConfig, work_dir: Path | None = None) -> None:
        super().__init__(work_dir)
        self.config = config

    def train(self, train_dataset: Dataset, validation_dataset: Dataset) -> Path:
        """
        Simulates training a potential.
        """
        logger.info("MockTrainer: Starting training...")

        if train_dataset.file_path:
            logger.info(f"MockTrainer: Training from file {train_dataset.file_path}")
        else:
            logger.info(f"MockTrainer: Training from memory ({len(train_dataset.structures)} items)")

        potential_filename = self.config.potential_output_name

        if self.work_dir:
            potential_path = self.work_dir / potential_filename
        else:
            # Fallback to CWD if work_dir not set (legacy behavior but discouraged)
            potential_path = Path(potential_filename)

        try:
            potential_path.write_text("Mock Potential Data")
            logger.info(f"MockTrainer: Wrote potential to {potential_path}")
        except OSError as e:
            msg = f"Failed to write mock potential file: {e}"
            logger.exception(msg)
            raise RuntimeError(msg) from e
        return potential_path

class MockValidator(BaseValidator):
    """
    Mock implementation of a Validator that returns metrics (randomized or config-based).
    """
    def __init__(self, config: ValidatorConfig) -> None:
        self.config = config

    def validate(self, potential_path: Path) -> ValidationResult:
        """
        Validates the potential by generating random metrics.
        """
        logger.info(f"MockValidator: Validating potential at {potential_path}")
        # Generate random metrics to be more realistic as requested
        rmse_energy = np.random.uniform(0.001, 0.1)
        rmse_forces = np.random.uniform(0.01, 0.5)

        return ValidationResult(
            metrics={"rmse_energy": rmse_energy, "rmse_forces": rmse_forces},
            is_stable=True
        )
