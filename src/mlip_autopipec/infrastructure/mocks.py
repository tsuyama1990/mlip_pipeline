import logging
import random
import uuid
from typing import Optional, List
from pathlib import Path
from ase import Atoms
from ase.io import write
import numpy as np

from mlip_autopipec.interfaces.oracle import BaseOracle
from mlip_autopipec.interfaces.trainer import BaseTrainer
from mlip_autopipec.interfaces.explorer import BaseExplorer
from mlip_autopipec.interfaces.validator import BaseValidator

from mlip_autopipec.domain_models.structure import Dataset, Structure
from mlip_autopipec.domain_models.potential import Potential, ExplorationResult
from mlip_autopipec.domain_models.validation import ValidationResult
from mlip_autopipec.config.config_model import OracleConfig, TrainerConfig, ExplorerConfig

logger = logging.getLogger(__name__)

class MockOracle(BaseOracle):
    def __init__(self, config: OracleConfig, work_dir: Path) -> None:
        self.config = config
        self.work_dir = work_dir

    def compute(self, dataset: Dataset) -> Dataset:
        logger.info(f"MockOracle computing energies for {len(dataset)} structures")
        for structure in dataset.structures:
            # Assign random energy
            energy = random.uniform(-100.0, -50.0) # noqa: S311
            structure.atoms.info["energy"] = energy
            structure.metadata["computed_by"] = "MockOracle"
        return dataset

class MockTrainer(BaseTrainer):
    def __init__(self, config: TrainerConfig, work_dir: Path) -> None:
        self.config = config
        self.work_dir = work_dir
        self.generation = 0

    def train(self, dataset: Dataset, previous_potential: Optional[Potential] = None) -> Potential:
        self.generation += 1
        logger.info(f"MockTrainer training generation {self.generation} with {len(dataset)} structures")

        pot_dir = self.work_dir / "potentials"
        pot_dir.mkdir(parents=True, exist_ok=True)

        pot_path = pot_dir / f"generation_{self.generation:03d}.yace"
        pot_path.write_text(f"Mock Potential Content Generation {self.generation}")

        return Potential(path=pot_path, name=f"mock_pot_{self.generation}")

class MockExplorer(BaseExplorer):
    def __init__(self, config: ExplorerConfig, work_dir: Path) -> None:
        self.config = config
        self.work_dir = work_dir
        self.steps = 0

    def explore(self, potential: Potential) -> ExplorationResult:
        self.steps += 1
        logger.info(f"MockExplorer exploring with potential {potential.name}, step {self.steps}")

        halted = True

        dump_path = self.work_dir / f"exploration_{self.steps}.xyz"

        atoms = Atoms('H2O', positions=[[0, 0, 0], [0, 1, 0], [0, 0, 1]])
        # Create list of atoms for trajectory
        traj: List[Atoms] = [atoms.copy() for _ in range(5)] # type: ignore[no-untyped-call]
        write(dump_path, traj)

        return ExplorationResult(
            halted=halted,
            dump_file=dump_path,
            high_gamma_frames=[2, 4] if halted else []
        )

class MockValidator(BaseValidator):
    def validate(self, potential: Potential) -> ValidationResult:
        logger.info(f"MockValidator validating {potential.name}")
        return ValidationResult(passed=True, metrics={"rmse": 0.0})
