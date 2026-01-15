# ruff: noqa: D101, D102
from unittest.mock import MagicMock

import pytest
from ase import Atoms

from mlip_autopipec.config_schemas import DFTConfig, DFTInput
from mlip_autopipec.modules.dft.exceptions import DFTCalculationError
from mlip_autopipec.modules.dft.factory import DFTFactory


@pytest.fixture
def mock_dft_config() -> DFTConfig:
    """Provide a mock DFTConfig for testing the DFTFactory."""
    dft_input = DFTInput(pseudopotentials={"Cu": "Cu.UPF"})
    config = DFTConfig(input=dft_input)
    # Populate with enough dummy adjustments for the tests
    config.retry_strategy.parameter_adjustments = [
        {"electrons.mixing_beta": 0.5},
        {"electrons.mixing_beta": 0.3},
        {"control.calculation": "'relax'"},
        {"electrons.mixing_beta": 0.1},
    ]
    return config


def test_dft_factory_success_on_first_try(
    mock_dft_config: DFTConfig, mocker: MagicMock
) -> None:
    """Test that the DFTFactory succeeds on the first attempt."""
    mocker.patch("mlip_autopipec.modules.dft.process_runner.QEProcessRunner.execute")
    mocker.patch(
        "mlip_autopipec.modules.dft.output_parser.QEOutputParser.parse",
        return_value={"energy": -1.0},
    )
    factory = DFTFactory(config=mock_dft_config)
    atoms = Atoms("Cu")
    result_atoms = factory.run(atoms)
    assert "energy" in result_atoms.calc.results
    assert result_atoms.calc.results["energy"] == -1.0


def test_dft_factory_retry_logic(mocker: MagicMock) -> None:
    """Test the retry logic of the DFTFactory."""
    mock_execute = mocker.patch(
        "mlip_autopipec.modules.dft.process_runner.QEProcessRunner.execute"
    )
    mock_execute.side_effect = [
        DFTCalculationError("L1 convergence failed"),
        None,  # Success on the second attempt
    ]
    mocker.patch(
        "mlip_autopipec.modules.dft.output_parser.QEOutputParser.parse",
        return_value={"energy": -1.0},
    )
    dft_input = DFTInput(pseudopotentials={"Cu": "Cu.UPF"})
    config = DFTConfig(input=dft_input)
    config.retry_strategy.max_retries = 1
    config.retry_strategy.parameter_adjustments = [{"electrons.mixing_beta": 0.5}]

    factory = DFTFactory(config=config)
    spy_generate = mocker.spy(factory.input_generator, "generate")

    atoms = Atoms("Cu")
    factory.run(atoms)

    assert mock_execute.call_count == 2
    assert spy_generate.call_count == 2
    second_call_config = spy_generate.call_args_list[1].kwargs["config"]
    assert second_call_config.input.electrons.mixing_beta == 0.5


def test_dft_factory_fails_after_all_retries(mocker: MagicMock) -> None:
    """Test that the DFTFactory raises an error after all retries fail."""
    mock_execute = mocker.patch(
        "mlip_autopipec.modules.dft.process_runner.QEProcessRunner.execute"
    )
    mock_execute.side_effect = [
        DFTCalculationError("L1"),
        DFTCalculationError("L2"),
    ]
    dft_input = DFTInput(pseudopotentials={"Cu": "Cu.UPF"})
    config = DFTConfig(input=dft_input)
    config.retry_strategy.max_retries = 1
    config.retry_strategy.parameter_adjustments = [{"electrons.mixing_beta": 0.1}]

    factory = DFTFactory(config=config)
    atoms = Atoms("Cu")

    with pytest.raises(DFTCalculationError):
        factory.run(atoms)

    assert mock_execute.call_count == 2
