from pathlib import Path

from mlip_autopipec.components.mock import (
    MockDynamics,
    MockGenerator,
    MockOracle,
    MockTrainer,
    MockValidator,
)
from mlip_autopipec.domain_models.config import (
    MockDynamicsConfig,
    MockGeneratorConfig,
    MockOracleConfig,
    MockTrainerConfig,
    MockValidatorConfig,
)
from mlip_autopipec.domain_models.inputs import ProjectState, Structure
from mlip_autopipec.domain_models.results import TrainingResult


def test_mock_generator(tmp_path: Path) -> None:
    """Test MockGenerator yields expected number of structures."""
    config = MockGeneratorConfig(type="mock", n_candidates=5)
    gen = MockGenerator(config, tmp_path)

    structures = list(gen.generate(ProjectState()))
    assert len(structures) == 5
    assert isinstance(structures[0], Structure)
    assert structures[0].tags["source"] == "mock_generator"


def test_mock_oracle(tmp_path: Path) -> None:
    """Test MockOracle computes properties."""
    config = MockOracleConfig(type="mock", noise_std=0.0)
    oracle = MockOracle(config, tmp_path)

    # Create dummy structure
    struct = Structure(
        positions=[[0, 0, 0]],
        numbers=[1],
        cell=[[10, 0, 0], [0, 10, 0], [0, 0, 10]],
        pbc=[True, True, True]
    )

    results = list(oracle.compute(iter([struct])))
    assert len(results) == 1
    res = results[0]

    assert "energy" in res.tags
    assert "forces" in res.tags
    assert "stress" in res.tags
    assert isinstance(res.tags["energy"], float)
    assert len(res.tags["forces"]) == 1


def test_mock_trainer(tmp_path: Path) -> None:
    """Test MockTrainer produces dummy potential."""
    config = MockTrainerConfig(type="mock")
    trainer = MockTrainer(config, tmp_path)

    dataset = tmp_path / "train.xyz"
    dataset.touch()

    res = trainer.train(dataset)
    assert isinstance(res, TrainingResult)
    assert res.potential_path.exists()
    assert res.metrics["rmse_energy"] == 0.005


def test_mock_dynamics(tmp_path: Path) -> None:
    """Test MockDynamics exploration."""
    config = MockDynamicsConfig(type="mock", steps=10)
    dyn = MockDynamics(config, tmp_path)

    initial = Structure(
        positions=[[0, 0, 0], [1, 0, 0]],
        numbers=[1, 1],
        cell=[[10, 0, 0], [0, 10, 0], [0, 0, 10]],
        pbc=[True, True, True]
    )
    potential = tmp_path / "pot.yace"
    potential.touch()

    results = list(dyn.explore(potential, initial))
    assert len(results) >= 1
    assert results[0].tags["provenance"] == "dynamics_halt"


def test_mock_validator(tmp_path: Path) -> None:
    """Test MockValidator returns passing metrics."""
    config = MockValidatorConfig(type="mock")
    validator = MockValidator(config, tmp_path)

    potential = tmp_path / "pot.yace"
    metrics = validator.validate(potential)

    assert metrics["phonon_stability"] is True
