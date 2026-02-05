from pathlib import Path

import pytest
from ase import Atoms

from mlip_autopipec.config.config_model import GlobalConfig
from mlip_autopipec.domain_models.dataset import Dataset
from mlip_autopipec.domain_models.structures import StructureMetadata
from mlip_autopipec.orchestration.mocks import MockExplorer, MockOracle, MockTrainer, MockValidator


@pytest.fixture
def config(tmp_path: Path) -> GlobalConfig:
    return GlobalConfig(work_dir=tmp_path)

def test_mock_oracle_batching(config: GlobalConfig) -> None:
    oracle = MockOracle(config=config)
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
    # Using atoms.copy() which is untyped in ASE
    structures = [
        StructureMetadata(structure=atoms.copy(), source="test", generation_method="manual") # type: ignore[no-untyped-call]
        for _ in range(5)
    ]

    # Calculate batch
    results = oracle.calculate(structures)
    assert len(results) == 5
    for res in results:
        # Check that energy/forces are calculated
        # Note: We can't easily check .calc in Mock if it's not attaching a real ASE calc,
        # but the mock should populate generic info or attach a SinglePointCalculator
        assert res.structure.calc is not None
        # Accessing potential_energy should work
        assert res.structure.get_potential_energy() is not None # type: ignore[no-untyped-call]

def test_mock_explorer_generation(config: GlobalConfig) -> None:
    explorer = MockExplorer(config=config)
    candidates = explorer.generate_candidates()
    assert len(candidates) > 0
    assert isinstance(candidates[0], StructureMetadata)

def test_mock_trainer_dataset_usage(config: GlobalConfig) -> None:
    trainer = MockTrainer(config=config)
    dataset = Dataset(name="train")
    # Add dummy structure
    atoms = Atoms('H')
    dataset.add(StructureMetadata(structure=atoms, source="test", generation_method="manual"))

    potential = trainer.train(dataset)
    assert potential is not None
    # Check if potential file was created in work_dir (mock behavior)
    # The file name might be random or fixed, checking any .yace or .pot
    found = list(config.work_dir.glob("*.yace"))
    assert len(found) > 0

def test_mock_validator(config: GlobalConfig) -> None:
    validator = MockValidator(config=config)
    res = validator.validate(potential=Path("dummy_pot"))
    assert res.passed is True
    assert len(res.metrics) > 0
