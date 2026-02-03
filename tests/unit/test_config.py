import pytest
from pydantic import ValidationError

from mlip_autopipec.config.config_model import (
    DFTConfig,
    ExplorationConfig,
    SimulationConfig,
    TrainingConfig,
)


def test_dft_config_valid() -> None:
    config = DFTConfig(code="qe", ecutwfc=40.0, kpoints=[2, 2, 2])
    assert config.code == "qe"
    assert config.ecutwfc == 40.0
    assert config.kpoints == [2, 2, 2]


def test_dft_config_invalid() -> None:
    with pytest.raises(ValidationError):
        DFTConfig(code="invalid", ecutwfc=40.0, kpoints=[2, 2, 2])  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        DFTConfig(code="qe", ecutwfc=-10.0, kpoints=[2, 2, 2])

    with pytest.raises(ValidationError):
        DFTConfig(code="qe", ecutwfc=40.0, kpoints=[2, 2])


def test_simulation_config_valid() -> None:
    config = SimulationConfig(
        project_name="Test",
        dft=DFTConfig(code="qe", ecutwfc=30.0, kpoints=[1, 1, 1]),
        training=TrainingConfig(code="pacemaker", cutoff=4.0),
        exploration=ExplorationConfig(strategy="random", steps=5),
    )
    assert config.project_name == "Test"
    assert config.dft.code == "qe"
    assert config.training.cutoff == 4.0
    assert config.exploration.steps == 5


def test_simulation_config_defaults() -> None:
    config = SimulationConfig(
        project_name="Test",
        dft=DFTConfig(code="qe", ecutwfc=30.0, kpoints=[1, 1, 1]),
        training=TrainingConfig(code="pacemaker", cutoff=4.0),
    )
    # Check default exploration
    assert config.exploration.strategy == "random"
    assert config.exploration.steps == 10


def test_simulation_config_extra_forbid() -> None:
    with pytest.raises(ValidationError):
        SimulationConfig(
            project_name="Test",
            dft=DFTConfig(code="qe", ecutwfc=30.0, kpoints=[1, 1, 1]),
            training=TrainingConfig(code="pacemaker", cutoff=4.0),
            extra_field="fail",  # type: ignore[call-arg]
        )
