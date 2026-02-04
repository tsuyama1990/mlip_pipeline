import logging
import time
from pathlib import Path

from ase.build import bulk
from ase.calculators.lj import LennardJones

from mlip_autopipec.config.config_model import ExplorationConfig, TrainingConfig
from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.structures import StructureMetadata
from mlip_autopipec.domain_models.validation import MetricResult, ValidationResult

logger = logging.getLogger(__name__)


class MockExplorer:
    def generate_candidates(self, config: ExplorationConfig) -> list[StructureMetadata]:
        candidates = []
        for _ in range(config.num_candidates):
            atoms = bulk("Cu", "fcc", a=3.6)
            atoms.rattle(stdev=0.1)  # type: ignore[no-untyped-call]
            meta = StructureMetadata(
                atoms=atoms, source="mock_explorer", generation_method="rattle", uncertainty=0.5
            )
            candidates.append(meta)
        logger.info(f"Generated {len(candidates)} candidate structures")
        return candidates


class MockOracle:
    def compute(self, structures: list[StructureMetadata]) -> list[StructureMetadata]:
        calc = LennardJones()  # type: ignore[no-untyped-call]
        for s in structures:
            s.atoms.calc = calc
            # Force calculation to populate atoms object
            s.atoms.get_forces()  # type: ignore[no-untyped-call]
            s.atoms.get_potential_energy()  # type: ignore[no-untyped-call]
        logger.info(f"Computed forces for {len(structures)} structures")
        return structures


class MockTrainer:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.cycle = 0

    def train(
        self, dataset: list[StructureMetadata], initial_potential: Potential | None = None
    ) -> Potential:
        self.cycle += 1
        name = f"potential_{self.cycle:03d}.yace"
        # Create dummy file
        with Path(name).open("w") as f:
            f.write("mock potential content")

        time.sleep(0.1)
        logger.info(f"Potential trained: {name}")
        return Potential(
            name=name,
            potential_type="mock",
            version=f"v{self.cycle}",
            path=name,
            metadata={"training_set_size": len(dataset)},
        )


class MockValidator:
    def validate(self, potential: Potential) -> ValidationResult:
        return ValidationResult(
            passed=True,
            metrics=[MetricResult(name="rmse_energy", passed=True, score=0.001)],
            report_path=None,
        )
