from pathlib import Path

from ase import Atoms

from mlip_autopipec.config.config_model import ExplorerConfig, OracleConfig, TrainerConfig
from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.structure import Dataset, Structure
from mlip_autopipec.infrastructure.mocks import MockExplorer, MockOracle, MockTrainer, MockValidator


def test_mock_oracle(tmp_path: Path) -> None:
    config = OracleConfig(type="mock")
    oracle = MockOracle(config, work_dir=tmp_path)

    atoms = Atoms('H')
    structure = Structure(atoms=atoms)
    dataset = Dataset(structures=[structure])

    computed_dataset = oracle.compute(dataset)
    assert len(computed_dataset) == 1
    assert "energy" in computed_dataset.structures[0].atoms.info
    assert computed_dataset.structures[0].metadata["computed_by"] == "MockOracle"

def test_mock_trainer(tmp_path: Path) -> None:
    config = TrainerConfig(type="mock")
    trainer = MockTrainer(config, work_dir=tmp_path)

    dataset = Dataset(structures=[])
    potential = trainer.train(dataset)

    assert potential.path.exists()
    assert potential.path.parent == tmp_path / "potentials"
    assert "generation_001.yace" in potential.path.name

def test_mock_explorer(tmp_path: Path) -> None:
    config = ExplorerConfig(type="mock")
    explorer = MockExplorer(config, work_dir=tmp_path)

    potential = Potential(path=tmp_path / "dummy.yace")
    result = explorer.explore(potential)

    assert result.halted is True
    assert result.dump_file.exists()
    assert len(result.high_gamma_frames) > 0

def test_mock_validator(tmp_path: Path) -> None:
    validator = MockValidator()
    potential = Potential(path=tmp_path / "dummy.yace")
    result = validator.validate(potential)

    assert result.passed is True
    assert "rmse" in result.metrics
