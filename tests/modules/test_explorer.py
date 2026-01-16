"""Unit tests for the SurrogateExplorer module."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from ase import Atoms

from mlip_autopipec.config_schemas import ExplorerParams, FPSParams, SurrogateModelParams
from mlip_autopipec.modules.descriptors import SOAPDescriptorCalculator
from mlip_autopipec.modules.explorer import SurrogateExplorer
from mlip_autopipec.modules.screening import SurrogateModelScreener


@pytest.fixture
def mock_explorer_config() -> ExplorerParams:
    """Fixture for creating a mock ExplorerParams object."""
    return ExplorerParams(
        surrogate_model=SurrogateModelParams(model_path="test_model.pt", energy_threshold_ev=-100.0),
        fps=FPSParams(num_structures_to_select=2)
    )


@pytest.fixture
def mock_dependencies() -> tuple[MagicMock, MagicMock]:
    """Fixture for creating mock dependencies for SurrogateExplorer."""
    return MagicMock(spec=SOAPDescriptorCalculator), MagicMock(spec=SurrogateModelScreener)


def test_explorer_initialization(
    mock_explorer_config: ExplorerParams,
    mock_dependencies: tuple[MagicMock, MagicMock],
) -> None:
    """Test that the SurrogateExplorer initializes correctly."""
    calc, screener = mock_dependencies
    explorer = SurrogateExplorer(
        config=mock_explorer_config, descriptor_calculator=calc, screener=screener
    )
    assert explorer.config == mock_explorer_config
    assert explorer.descriptor_calculator == calc
    assert explorer.screener == screener


def test_select_returns_empty_list_for_empty_input(
    mock_explorer_config: ExplorerParams,
    mock_dependencies: tuple[MagicMock, MagicMock],
) -> None:
    """Test that the select method returns an empty list for empty input."""
    calc, screener = mock_dependencies
    explorer = SurrogateExplorer(
        config=mock_explorer_config, descriptor_calculator=calc, screener=screener
    )
    assert explorer.select([]) == []


def test_select_returns_all_candidates_if_less_than_selection_num(
    mock_explorer_config: ExplorerParams,
    mock_dependencies: tuple[MagicMock, MagicMock],
) -> None:
    """Test that all candidates are returned if their number is less than the selection number."""
    calc, screener = mock_dependencies
    candidates = [Atoms("H"), Atoms("He")]
    screener.screen.return_value = candidates
    explorer = SurrogateExplorer(
        config=mock_explorer_config, descriptor_calculator=calc, screener=screener
    )
    assert explorer.select(candidates) == candidates


def test_select_returns_empty_list_if_all_screened_out(
    mock_explorer_config: ExplorerParams,
    mock_dependencies: tuple[MagicMock, MagicMock],
) -> None:
    """Test that an empty list is returned if all candidates are screened out."""
    calc, screener = mock_dependencies
    screener.screen.return_value = []
    explorer = SurrogateExplorer(
        config=mock_explorer_config, descriptor_calculator=calc, screener=screener
    )
    assert explorer.select([Atoms("H")]) == []


def test_farthest_point_sampling_selects_correct_number(
    mock_explorer_config: ExplorerParams,
    mock_dependencies: tuple[MagicMock, MagicMock],
) -> None:
    """Test that Farthest Point Sampling selects the correct number of structures."""
    calc, screener = mock_dependencies
    explorer = SurrogateExplorer(
        config=mock_explorer_config, descriptor_calculator=calc, screener=screener
    )
    descriptors = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    selected_indices = explorer._farthest_point_sampling(descriptors, 2)
    assert len(selected_indices) == 2
