import numpy as np
import secrets
from pathlib import Path
from typing import Iterator, Iterable, Any

from mlip_autopipec.domain_models import Structure, Potential
from mlip_autopipec.interfaces import BaseGenerator, BaseOracle, BaseTrainer, BaseDynamics

class MockGenerator(BaseGenerator):
    def __init__(self, **kwargs: Any) -> None:
        self.rng = np.random.default_rng(secrets.randbits(128))

    def generate(self, count: int) -> Iterator[Structure]:
        for _ in range(count):
            yield Structure(
                atomic_numbers=[1, 1],
                positions=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                cell=[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                pbc=[True, True, True],
            )

class MockOracle(BaseOracle):
    def __init__(self, **kwargs: Any) -> None:
        self.rng = np.random.default_rng(secrets.randbits(128))

    def compute(self, structures: Iterable[Structure]) -> Iterator[Structure]:
        for s in structures:
            new_s = s.model_copy(deep=True)
            # Assign random energy and forces
            new_s.energy = float(self.rng.uniform(-10.0, -5.0))
            new_s.forces = self.rng.uniform(-0.1, 0.1, size=(len(s.atomic_numbers), 3)).tolist()
            new_s.stress = self.rng.uniform(-0.1, 0.1, size=(3, 3)).tolist()
            yield new_s

class MockTrainer(BaseTrainer):
    def __init__(self, **kwargs: Any) -> None:
        pass

    def train(self, dataset: Iterable[Structure], workdir: Path) -> Potential:
        # Create a dummy potential file
        pot_path = workdir / "potential.yace"
        # Ensure parent directory exists
        pot_path.parent.mkdir(parents=True, exist_ok=True)
        pot_path.touch()
        return Potential(path=pot_path, format="yace", description="Mock potential")

class MockDynamics(BaseDynamics):
    def __init__(self, **kwargs: Any) -> None:
        pass

    def explore(self, potential: Potential) -> Iterator[Structure]:
        # Return a few structures
        for _ in range(2):
            yield Structure(
                atomic_numbers=[1, 1],
                positions=[[0.0, 0.0, 0.0], [1.1, 1.1, 1.1]], # Slightly different
                cell=[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                pbc=[True, True, True],
                properties={"uncertainty": 0.5} # High uncertainty
            )
