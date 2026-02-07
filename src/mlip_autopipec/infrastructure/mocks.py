import logging
import secrets
import tempfile
from pathlib import Path

import numpy as np

from mlip_autopipec.domain_models import (
    ExplorationResult,
    Potential,
    Structure,
    ValidationResult,
)
from mlip_autopipec.interfaces import (
    BaseDynamics,
    BaseOracle,
    BaseSelector,
    BaseStructureGenerator,
    BaseTrainer,
    BaseValidator,
)

logger = logging.getLogger(__name__)


class MockOracle(BaseOracle):
    def compute(self, structures: list[Structure]) -> list[Structure]:
        logger.info(f"MockOracle computing for {len(structures)} structures")
        rng = np.random.default_rng(secrets.randbits(128))
        results = []
        for s in structures:
            new_s = s.model_copy(deep=True)
            n_atoms = len(new_s.symbols)
            # Add fake energy, forces, stress
            new_s.properties["energy"] = rng.uniform(-100.0, 0.0)
            new_s.properties["forces"] = rng.uniform(-1.0, 1.0, (n_atoms, 3))
            new_s.properties["stress"] = rng.uniform(-1.0, 1.0, (3, 3))
            results.append(new_s)
        return results


class MockTrainer(BaseTrainer):
    def train(self, structures: list[Structure], workdir: Path) -> Potential:
        logger.info(f"MockTrainer training on {len(structures)} structures")

        # Strict workdir validation
        workdir = workdir.resolve()
        try:
            workdir.relative_to(Path.cwd())
        except ValueError:
            try:
                workdir.relative_to(Path(tempfile.gettempdir()))
            except ValueError as e:
                msg = f"Workdir {workdir} must be in CWD or tempdir"
                raise ValueError(msg) from e

        workdir.mkdir(parents=True, exist_ok=True)
        dummy_file = workdir / "dummy.yace"
        dummy_file.touch()

        return Potential(
            path=dummy_file,
            version="mock_v1",
            metrics={"rmse_energy": 0.01, "rmse_forces": 0.1},
        )


class MockDynamics(BaseDynamics):
    def run(self, potential: Potential, structure: Structure) -> ExplorationResult:
        logger.info("MockDynamics running")
        rng = np.random.default_rng(secrets.randbits(128))

        halt_prob = float(self.params.get("halt_probability", 0.5))

        if rng.random() < halt_prob:
            status = "halted"
            # Return a perturbed structure as the "halt structure"
            final_structure = structure.model_copy(deep=True)
            final_structure.positions += rng.uniform(-0.1, 0.1, final_structure.positions.shape)
        else:
            status = "converged"
            # Return "relaxed" structure
            final_structure = structure.model_copy(deep=True)
            final_structure.positions += rng.uniform(-0.01, 0.01, final_structure.positions.shape)

        return ExplorationResult(
            status=status,  # type: ignore[arg-type]
            structure=final_structure,
            metrics={"steps": 100},
        )


class MockStructureGenerator(BaseStructureGenerator):
    def generate(self, structure: Structure) -> list[Structure]:
        logger.info("MockGenerator generating")
        rng = np.random.default_rng(secrets.randbits(128))

        n_candidates = int(self.params.get("n_candidates", 5))
        candidates = []
        for _ in range(n_candidates):
            new_s = structure.model_copy(deep=True)
            new_s.positions += rng.uniform(-0.5, 0.5, new_s.positions.shape)
            candidates.append(new_s)
        return candidates


class MockValidator(BaseValidator):
    def validate(self, potential: Potential) -> ValidationResult:
        logger.info("MockValidator validating")
        return ValidationResult(
            passed=True,
            metrics={"validation_error": 0.02},
            details={"comment": "Mock validation passed"},
        )


class MockSelector(BaseSelector):
    def select(self, candidates: list[Structure], n: int) -> list[Structure]:
        logger.info(f"MockSelector selecting {n} from {len(candidates)}")
        return candidates[:n]
