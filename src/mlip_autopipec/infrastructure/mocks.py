import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from mlip_autopipec.domain_models.potential import ExplorationResult, Potential
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.validation import ValidationResult
from mlip_autopipec.interfaces import (
    BaseDynamics,
    BaseOracle,
    BaseStructureGenerator,
    BaseTrainer,
    BaseValidator,
)


class MockOracle(BaseOracle):
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}

    def compute(
        self, structures: list[Structure], workdir: str | Path = Path()
    ) -> list[Structure]:
        time.sleep(0.1)
        for s in structures:
            s.energy = random.uniform(-100.0, -10.0)  # noqa: S311
            s.forces = np.random.uniform(
                -1.0, 1.0, size=(len(s.species), 3)
            ).tolist()
            s.stress = np.random.uniform(
                -0.1, 0.1, size=(3, 3)
            ).tolist()
        return structures


class MockTrainer(BaseTrainer):
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}

    def train(
        self, structures: list[Structure], workdir: str | Path = Path()
    ) -> Potential:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        potential_path = workdir / "potential.yace"
        potential_path.touch()
        with (workdir / "training_report.txt").open("w") as f:
            f.write("Training completed successfully (Mock).")
        return Potential(path=str(potential_path.absolute()))


class MockDynamics(BaseDynamics):
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}

    def run_exploration(
        self, potential: Potential, workdir: str | Path = Path()
    ) -> ExplorationResult:
        # Use param to control halt probability
        # Default to 1.0 (always halt) to ensure loop continues in mock mode by default
        halt_prob = self.params.get("halt_prob", 1.0)
        halted = random.random() < halt_prob  # noqa: S311
        structures = []
        if halted:
            structures = [
                Structure(
                    positions=[[0.0, 0.0, 0.0]],
                    cell=[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                    species=["H"],
                )
            ]
        return ExplorationResult(halted=halted, structures=structures)


class MockStructureGenerator(BaseStructureGenerator):
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}

    def get_candidates(self) -> list[Structure]:
        return [
            Structure(
                positions=np.random.rand(2, 3) * 10,
                cell=np.eye(3) * 10,
                species=["Fe", "Fe"],
            )
            for _ in range(5)
        ]


class MockValidator(BaseValidator):
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}

    def validate(
        self, potential: Potential, workdir: str | Path = Path()
    ) -> ValidationResult:
        # Default to False to ensure loop continues
        passed = self.params.get("force_pass", False)
        return ValidationResult(
            passed=passed, metrics={"rmse_energy": 0.01, "rmse_forces": 0.05}
        )
