from unittest.mock import MagicMock

import numpy as np
import pytest
from ase import Atoms
from pydantic import ValidationError

from mlip_autopipec.modules.b_explorer import (
    SurrogateExplorer,
    farthest_point_sampling,
)
from mlip_autopipec.schemas.user_config import SurrogateConfig


def test_farthest_point_sampling() -> None:
    """Test the FPS algorithm with a simple 2D point set."""
    np.random.seed(42)
    points = np.array(
        [
            [0, 0],
            [0, 1],
            [0, 2],
            [1, 2],
            [2, 2],
            [2, 1],
            [2, 0],
            [1, 0],
            [1, 1],
        ]
    )
    num_to_select = 4
    selected_indices = farthest_point_sampling(points, num_to_select)

    # The four corners should be selected
    expected_corners = {0, 2, 4, 6}
    assert set(selected_indices) == expected_corners


def test_surrogate_explorer_selection(mocker: "MagicMock") -> None:
    """Test the main selection logic of the SurrogateExplorer."""
    # Mock the MACE calculator
    mock_calculator = mocker.MagicMock()
    mock_calculator.get_potential_energy.side_effect = [
        -10.0,
        -11.0,
        -10.5,
        -1000.0,
        -9.0,
    ]
    mocker.patch(
        "mlip_autopipec.modules.b_explorer.mace_mp", return_value=mock_calculator
    )

    # Mock dscribe.SOAP
    mock_soap = mocker.MagicMock()
    mock_soap.create.return_value = np.random.rand(5, 10)  # 5 structures, 10 features
    mocker.patch("dscribe.descriptors.SOAP", return_value=mock_soap)

    config = SurrogateConfig(
        model_path="mace_mp_small.model", num_to_select_fps=3, descriptor_type="SOAP"
    )
    explorer = SurrogateExplorer(config)

    structures = [
        Atoms("H", positions=[(0, 0, 0)], cell=[1, 1, 1], pbc=True) for _ in range(5)
    ]
    selected_structures = explorer.select_structures(structures)

    # One structure should be filtered out due to high energy
    assert len(selected_structures) == 3

    # Check that MACE was called for all initial structures
    assert mock_calculator.get_potential_energy.call_count == 5


def test_surrogate_config_validation() -> None:
    """Test Pydantic validation for SurrogateConfig."""
    # Valid config
    SurrogateConfig(model_path="path/to/model", num_to_select_fps=10, descriptor_type="SOAP")

    # Invalid num_to_select_fps
    with pytest.raises(ValidationError):
        SurrogateConfig(
            model_path="path/to/model",
            num_to_select_fps=-1,
            descriptor_type="SOAP",
        )
