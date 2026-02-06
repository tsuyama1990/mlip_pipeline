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
    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir

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

        # Write to file
        candidates_file = self.work_dir / f"candidates_{uuid.uuid4().hex}.xyz"
        count = 0
        for meta in _generate():
            write(candidates_file, meta.structure, append=True)
            count += 1

        logger.info(f"MockExplorer: Generated {count} structures at {candidates_file}.")
        return Dataset(file_path=candidates_file)

class MockOracle(BaseOracle):
    """
    Mock implementation of an Oracle that assigns random energies and forces.
    """
    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir

    def label(self, dataset: Dataset) -> Dataset:
        """
        Labels the dataset with random physical properties.
        """
        logger.info("MockOracle: Labeling structures...")

        def _label_structure(meta: StructureMetadata) -> StructureMetadata:
            if meta.structure is None:
                logger.warning("MockOracle: Encountered StructureMetadata with None structure.")
                return meta

            # Simulate labeling
            meta.energy = np.random.uniform(-100, -10)
            n_atoms = len(meta.structure)
            meta.forces = np.random.uniform(-1, 1, size=(n_atoms, 3)).tolist()
            meta.virial = np.random.uniform(-0.1, 0.1, size=(3, 3)).tolist()
            return meta

        if dataset.file_path:
            logger.info(f"MockOracle: Streaming from file {dataset.file_path}")
            output_file = self.work_dir / f"labeled_{uuid.uuid4().hex}.xyz"

            count = 0
            # Streaming read -> Label -> Streaming write
            for atoms in iread(dataset.file_path):
                meta = StructureMetadata(structure=atoms)
                labeled_meta = _label_structure(meta)
                count += 1

                # Write back
                write(output_file, labeled_meta.structure, append=True)

            logger.info(f"MockOracle: Labeled {count} structures and wrote to {output_file}")
            return Dataset(file_path=output_file)

        msg = "MockOracle only supports file-based Datasets"
        raise ValueError(msg)

class MockTrainer(BaseTrainer):
    """
    Mock implementation of a Trainer that produces a dummy potential file.
    """
    def __init__(self, config: TrainerConfig, work_dir: Path) -> None:
        self.config = config
        self.work_dir = work_dir

    def train(self, train_dataset: Dataset, validation_dataset: Dataset) -> Path:
        """
        Simulates training a potential.
        """
        logger.info("MockTrainer: Starting training...")
        logger.info(f"MockTrainer: Training from file {train_dataset.file_path}")

        potential_filename = self.config.potential_output_name
        potential_path = self.work_dir / potential_filename

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
        logger.info(f"MockValidator: Validating potential at {potential_path}")
        # Generate random metrics to be more realistic as requested
        rmse_energy = np.random.uniform(0.001, 0.1)
        rmse_forces = np.random.uniform(0.01, 0.5)

        return ValidationResult(
            metrics={"rmse_energy": rmse_energy, "rmse_forces": rmse_forces},
            is_stable=True
        )
