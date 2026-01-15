# ruff: noqa: D101, D102
import pytest
from ase import Atoms

from mlip_autopipec.config_schemas import SystemConfig
from mlip_autopipec.modules.inference import LammpsRunner, UncertaintyQuantifier


@pytest.fixture
def mock_system_config() -> SystemConfig:
    """Provide a mock SystemConfig for testing the LammpsRunner."""
    config_dict = {
        "target_system": {"elements": ["Cu"], "composition": {"Cu": 1.0}},
        "dft": {
            "executable": {"command": "pw.x"},
            "input": {"pseudopotentials": {"Cu": "Cu.UPF"}},
        },
        "inference": {
            "uncertainty_threshold": 4.0,
            "total_simulation_steps": 10,
        },
    }
    return SystemConfig(**config_dict)


def test_lammps_runner_initialization(mock_system_config: SystemConfig) -> None:
    """Test that the LammpsRunner initializes correctly."""
    quantifier = UncertaintyQuantifier()
    runner = LammpsRunner(
        config=mock_system_config, potential_path="test.yace", quantifier=quantifier
    )
    assert runner.config == mock_system_config
    assert runner.potential_path == "test.yace"
    assert runner.quantifier == quantifier


def test_runner_raises_error_if_inference_config_missing(
    mock_system_config: SystemConfig,
) -> None:
    """Test that ValueError is raised if inference config is missing."""
    mock_system_config.inference = None
    with pytest.raises(
        ValueError, match="Inference parameters must be defined in the config."
    ):
        LammpsRunner(
            config=mock_system_config,
            potential_path="test.yace",
            quantifier=UncertaintyQuantifier(),
        )


def test_runner_yields_on_uncertainty(mock_system_config: SystemConfig) -> None:
    """Verify the core generator behavior of the LammpsRunner."""
    quantifier = UncertaintyQuantifier()
    quantifier._mock_sequence = [1.0, 1.5, 2.0, 4.5, 2.5]  # Threshold is 4.0
    runner = LammpsRunner(
        config=mock_system_config, potential_path="test.yace", quantifier=quantifier
    )
    generator = runner.run()

    assert next(generator) == "stable"
    assert next(generator) == "stable"
    assert next(generator) == "stable"
    assert isinstance(next(generator), Atoms)
    assert next(generator) == "stable"


def test_runner_stops_at_total_steps(mock_system_config: SystemConfig) -> None:
    """Test that the simulation stops after the specified number of steps."""
    assert mock_system_config.inference is not None  # For type checker
    mock_system_config.inference.total_simulation_steps = 5
    quantifier = UncertaintyQuantifier()
    quantifier._mock_sequence = [1.0] * 10
    runner = LammpsRunner(
        config=mock_system_config, potential_path="test.yace", quantifier=quantifier
    )
    generator = runner.run()

    for _ in range(5):
        next(generator)

    with pytest.raises(StopIteration):
        next(generator)
