# ruff: noqa: D101
"""Tests for the SurrogateExplorer module."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from ase import Atoms
from pytest_mock import MockerFixture

from mlip_autopipec.config_schemas import (
    ExplorerParams,
    FPSParams,
    SOAPParams,
    SurrogateModelParams,
)
from mlip_autopipec.modules.descriptors import SOAPDescriptorCalculator
from mlip_autopipec.modules.explorer import SurrogateExplorer


@pytest.fixture
def mock_explorer_config() -> ExplorerParams:
    """Create a mock ExplorerParams for testing."""
    return ExplorerParams(
        surrogate_model=SurrogateModelParams(
            model_path="dummy/path/to/model.model", energy_threshold_ev=0.0
        ),
        fps=FPSParams(num_structures_to_select=3, soap_params=SOAPParams()),
    )


@pytest.fixture
def mock_descriptor_calculator(mocker: MockerFixture) -> MagicMock:
    """Create a mock SOAPDescriptorCalculator."""
    mock: MagicMock = mocker.MagicMock(spec=SOAPDescriptorCalculator)
    mock.calculate.return_value = np.array([])  # Default return
    return mock


def test_fps_selection_logic(
    mock_explorer_config: ExplorerParams,
    mock_descriptor_calculator: MagicMock,
) -> None:
    """Unit test the Farthest Point Sampling algorithm's core logic."""
    descriptors = np.array([[0, 0], [0.1, 0.1], [0.2, 0], [10, 10], [20, 0], [0, 20]])
    num_structures_to_select = 3

    explorer = SurrogateExplorer(mock_explorer_config, mock_descriptor_calculator)
    np.random.seed(0)
    selected_indices = explorer._farthest_point_sampling(
        descriptors, num_structures_to_select
    )

    assert sorted(selected_indices) == [0, 4, 5]


def test_select_pipeline(
    mocker: MockerFixture,
    mock_explorer_config: ExplorerParams,
    mock_descriptor_calculator: MagicMock,
) -> None:
    """Integration test for the full select pipeline using mocks."""
    candidates = [Atoms("H", positions=[(0, 0, i)]) for i in range(10)]
    mock_explorer_config.surrogate_model.energy_threshold_ev = -5.0
    mock_explorer_config.fps.num_structures_to_select = 3

    mock_mace = mocker.patch("mlip_autopipec.modules.explorer.mace_mp")
    mock_mace.return_value.get_potential_energy.side_effect = [-10.0] * 7 + [1.0] * 3

    mock_descriptor_calculator.calculate.return_value = np.random.rand(7, 10)

    explorer = SurrogateExplorer(mock_explorer_config, mock_descriptor_calculator)
    np.random.seed(42)
    final_selection = explorer.select(candidates)

    assert len(final_selection) == 3
    final_indices = {c.positions[0, 2] for c in final_selection}
    assert final_indices.issubset(set(range(7)))

    mock_mace.assert_called_once_with(
        model="dummy/path/to/model.model", device="cpu", default_dtype="float64"
    )
    mock_descriptor_calculator.calculate.assert_called_once()
    assert len(mock_descriptor_calculator.calculate.call_args[0][0]) == 7


def test_empty_candidate_list(
    mock_explorer_config: ExplorerParams,
    mock_descriptor_calculator: MagicMock,
) -> None:
    """Test that an empty candidate list is handled gracefully."""
    explorer = SurrogateExplorer(mock_explorer_config, mock_descriptor_calculator)
    final_selection = explorer.select([])
    assert final_selection == []


def test_all_candidates_screened(
    mocker: MockerFixture,
    mock_explorer_config: ExplorerParams,
    mock_descriptor_calculator: MagicMock,
) -> None:
    """Test that if all candidates are screened out, an empty list is returned."""
    candidates = [Atoms("H")] * 5
    mock_explorer_config.surrogate_model.energy_threshold_ev = -5.0
    mocker.patch(
        "mlip_autopipec.modules.explorer.mace_mp"
    ).return_value.get_potential_energy.return_value = 1.0

    explorer = SurrogateExplorer(mock_explorer_config, mock_descriptor_calculator)
    final_selection = explorer.select(candidates)
    assert final_selection == []
    mock_descriptor_calculator.calculate.assert_not_called()


def test_fewer_candidates_than_num_select(
    mocker: MockerFixture,
    mock_explorer_config: ExplorerParams,
    mock_descriptor_calculator: MagicMock,
) -> None:
    """Test that if fewer candidates remain than num_select, all are returned."""
    candidates = [Atoms("H")] * 5
    mock_explorer_config.surrogate_model.energy_threshold_ev = -5.0
    mock_explorer_config.fps.num_structures_to_select = 10
    mocker.patch(
        "mlip_autopipec.modules.explorer.mace_mp"
    ).return_value.get_potential_energy.return_value = -10.0

    explorer = SurrogateExplorer(mock_explorer_config, mock_descriptor_calculator)
    final_selection = explorer.select(candidates)
    assert len(final_selection) == 5
    mock_descriptor_calculator.calculate.assert_not_called()


def test_invalid_energy_values(
    mocker: MockerFixture,
    mock_explorer_config: ExplorerParams,
    mock_descriptor_calculator: MagicMock,
) -> None:
    """Test that NaN and Inf energy values are handled correctly."""
    candidates = [Atoms("H")] * 3
    mock_explorer_config.surrogate_model.energy_threshold_ev = -5.0
    mocker.patch(
        "mlip_autopipec.modules.explorer.mace_mp"
    ).return_value.get_potential_energy.side_effect = [-10.0, np.nan, np.inf]

    explorer = SurrogateExplorer(mock_explorer_config, mock_descriptor_calculator)
    final_selection = explorer.select(candidates)
    assert len(final_selection) == 1
