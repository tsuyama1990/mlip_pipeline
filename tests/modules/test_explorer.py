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
from mlip_autopipec.modules.screening import SurrogateModelScreener


@pytest.fixture
def mock_surrogate_explorer_config() -> ExplorerParams:
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


@pytest.fixture
def mock_screener(mocker: MockerFixture) -> MagicMock:
    """Create a mock SurrogateModelScreener."""
    mock: MagicMock = mocker.MagicMock(spec=SurrogateModelScreener)
    mock.screen.return_value = []  # Default return
    return mock


def test_fps_selection_logic(
    mock_surrogate_explorer_config: ExplorerParams,
    mock_descriptor_calculator: MagicMock,
    mock_screener: MagicMock,
) -> None:
    """Unit test the Farthest Point Sampling algorithm's core logic."""
    descriptors = np.array([[0, 0], [0.1, 0.1], [0.2, 0], [10, 10], [20, 0], [0, 20]])
    num_structures_to_select = 3

    explorer = SurrogateExplorer(
        mock_surrogate_explorer_config, mock_descriptor_calculator, mock_screener
    )
    np.random.seed(0)
    selected_indices = explorer._farthest_point_sampling(descriptors, num_structures_to_select)

    assert sorted(selected_indices) == [0, 4, 5]


def test_select_pipeline(
    mocker: MockerFixture,
    mock_surrogate_explorer_config: ExplorerParams,
    mock_descriptor_calculator: MagicMock,
    mock_screener: MagicMock,
) -> None:
    """Integration test for the full select pipeline using mocks."""
    candidates = [Atoms("H", positions=[(0, 0, i)]) for i in range(10)]
    mock_surrogate_explorer_config.fps.num_structures_to_select = 3

    # Configure the mock screener to return 7 out of 10 candidates
    mock_screener.screen.return_value = candidates[:7]
    mock_descriptor_calculator.calculate.return_value = np.random.rand(7, 10)

    explorer = SurrogateExplorer(
        mock_surrogate_explorer_config, mock_descriptor_calculator, mock_screener
    )
    np.random.seed(42)
    final_selection = explorer.select(candidates)

    assert len(final_selection) == 3
    final_indices = {c.positions[0, 2] for c in final_selection}
    assert final_indices.issubset(set(range(7)))

    mock_screener.screen.assert_called_once_with(candidates)
    mock_descriptor_calculator.calculate.assert_called_once_with(candidates[:7])


def test_empty_candidate_list(
    mock_surrogate_explorer_config: ExplorerParams,
    mock_descriptor_calculator: MagicMock,
    mock_screener: MagicMock,
) -> None:
    """Test that an empty candidate list is handled gracefully."""
    explorer = SurrogateExplorer(
        mock_surrogate_explorer_config, mock_descriptor_calculator, mock_screener
    )
    final_selection = explorer.select([])
    assert final_selection == []
    mock_screener.screen.assert_not_called()


def test_all_candidates_screened(
    mock_surrogate_explorer_config: ExplorerParams,
    mock_descriptor_calculator: MagicMock,
    mock_screener: MagicMock,
) -> None:
    """Test that if all candidates are screened out, an empty list is returned."""
    candidates = [Atoms("H")] * 5
    mock_screener.screen.return_value = []

    explorer = SurrogateExplorer(
        mock_surrogate_explorer_config, mock_descriptor_calculator, mock_screener
    )
    final_selection = explorer.select(candidates)
    assert final_selection == []
    mock_descriptor_calculator.calculate.assert_not_called()


def test_fewer_candidates_than_num_select(
    mock_surrogate_explorer_config: ExplorerParams,
    mock_descriptor_calculator: MagicMock,
    mock_screener: MagicMock,
) -> None:
    """Test that if fewer candidates remain than num_select, all are returned."""
    candidates = [Atoms("H")] * 5
    mock_surrogate_explorer_config.fps.num_structures_to_select = 10
    mock_screener.screen.return_value = candidates

    explorer = SurrogateExplorer(
        mock_surrogate_explorer_config, mock_descriptor_calculator, mock_screener
    )
    final_selection = explorer.select(candidates)
    assert len(final_selection) == 5
    mock_descriptor_calculator.calculate.assert_not_called()
