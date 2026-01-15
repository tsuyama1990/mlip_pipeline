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


def test_dft_factory_succeeds_on_first_try(
    mock_dft_config: DFTConfig, mocker: MagicMock
) -> None:
    """Verify the factory returns a correct result when the first attempt succeeds."""
    atoms = Atoms("Cu")
    atoms_with_results = atoms.copy()  # type: ignore[no-untyped-call]
    atoms_with_results.calc = MagicMock(results={"energy": -1.0})

    factory = DFTFactory(config=mock_dft_config)
    mock_run_single = mocker.patch.object(
        factory, "_run_single_calculation", return_value=atoms_with_results
    )

    result_atoms = factory.run(atoms)

    mock_run_single.assert_called_once_with(atoms)
    assert "energy" in result_atoms.calc.results
    assert result_atoms.calc.results["energy"] == -1.0


def test_dft_factory_succeeds_on_retry(
    mock_dft_config: DFTConfig, mocker: MagicMock
) -> None:
    """Verify the factory successfully retries after a failure."""
    atoms = Atoms("Cu")
    atoms_with_results = atoms.copy()  # type: ignore[no-untyped-call]
    atoms_with_results.calc = MagicMock(results={"energy": -2.0})

    mock_dft_config.retry_strategy.max_retries = 2
    factory = DFTFactory(config=mock_dft_config)

    # Mock the internal method to fail once, then succeed
    mock_run_single = mocker.patch.object(
        factory,
        "_run_single_calculation",
        side_effect=[DFTCalculationError("Convergence failed"), atoms_with_results],
    )

    result_atoms = factory.run(atoms)

    assert mock_run_single.call_count == 2
    assert "energy" in result_atoms.calc.results
    assert result_atoms.calc.results["energy"] == -2.0


def test_dft_factory_fails_after_exhausting_retries(
    mock_dft_config: DFTConfig, mocker: MagicMock
) -> None:
    """Verify the factory raises an error after all retries are exhausted."""
    atoms = Atoms("Cu")
    # Set max_retries to 2, meaning 1 initial call + 2 retries = 3 total calls
    mock_dft_config.retry_strategy.max_retries = 2
    factory = DFTFactory(config=mock_dft_config)

    # Mock the internal method to always fail
    mock_run_single = mocker.patch.object(
        factory,
        "_run_single_calculation",
        side_effect=DFTCalculationError("Persistent failure"),
    )

    with pytest.raises(DFTCalculationError, match="Persistent failure"):
        factory.run(atoms)

    # The number of calls should be 1 (initial) + max_retries
    total_expected_calls = 1 + mock_dft_config.retry_strategy.max_retries
    assert mock_run_single.call_count == total_expected_calls
