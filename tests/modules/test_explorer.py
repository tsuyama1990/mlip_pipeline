"""Unit tests for the Surrogate Explorer."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from ase import Atoms

from mlip_autopipec.config.models import ExplorerParams, FPSParams, SOAPParams, SurrogateModelParams
from mlip_autopipec.modules.descriptors import SOAPDescriptorCalculator
from mlip_autopipec.modules.explorer import SurrogateExplorer
from mlip_autopipec.modules.screening import SurrogateModelScreener


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    return ExplorerParams(
        surrogate_model=SurrogateModelParams(model_path="dummy.model"),
        fps=FPSParams(
            num_structures_to_select=2,
            soap_params=SOAPParams(n_max=2, l_max=2, r_cut=3.0, atomic_sigma=0.5),
        ),
    )


@pytest.fixture
def mock_descriptor_calculator():
    """Create a mock descriptor calculator."""
    mock = MagicMock(spec=SOAPDescriptorCalculator)
    # Mock return value for calculate (random features)
    mock.calculate.return_value = np.random.rand(5, 10)
    return mock


@pytest.fixture
def mock_screener():
    """Create a mock screener."""
    mock = MagicMock(spec=SurrogateModelScreener)
    return mock


def test_select_empty_list(mock_config, mock_descriptor_calculator, mock_screener):
    """Test selection with an empty candidate list."""
    explorer = SurrogateExplorer(mock_config, mock_descriptor_calculator, mock_screener)
    result = explorer.select([])
    assert result == []
    mock_screener.screen.assert_not_called()


def test_select_all_screened_out(mock_config, mock_descriptor_calculator, mock_screener):
    """Test when the screener filters out all candidates."""
    mock_screener.screen.return_value = []
    candidates = [Atoms("H"), Atoms("He")]

    explorer = SurrogateExplorer(mock_config, mock_descriptor_calculator, mock_screener)
    result = explorer.select(candidates)

    assert result == []
    mock_screener.screen.assert_called_once_with(candidates)
    mock_descriptor_calculator.calculate.assert_not_called()


def test_select_fewer_than_requested(mock_config, mock_descriptor_calculator, mock_screener):
    """Test when fewer candidates pass screening than requested by FPS."""
    # Requested is 2 in fixture
    candidates = [Atoms("H")]
    mock_screener.screen.return_value = candidates

    explorer = SurrogateExplorer(mock_config, mock_descriptor_calculator, mock_screener)
    result = explorer.select(candidates)

    assert result == candidates
    mock_screener.screen.assert_called_once()
    # Should skip FPS if not enough candidates
    mock_descriptor_calculator.calculate.assert_not_called()


def test_select_fps(mock_config, mock_descriptor_calculator, mock_screener):
    """Test normal FPS selection flow."""
    # 5 candidates, want 2
    candidates = [Atoms("H") for _ in range(5)]
    mock_screener.screen.return_value = candidates

    # Descriptors for 5 atoms
    mock_descriptor_calculator.calculate.return_value = np.random.rand(5, 10)

    explorer = SurrogateExplorer(mock_config, mock_descriptor_calculator, mock_screener)
    result = explorer.select(candidates)

    assert len(result) == 2
    mock_screener.screen.assert_called_once()
    mock_descriptor_calculator.calculate.assert_called_once()
