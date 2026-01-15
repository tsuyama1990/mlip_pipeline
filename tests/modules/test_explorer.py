# ruff: noqa: D101
"""Tests for the SurrogateExplorer module."""

import numpy as np
import pytest
from ase import Atoms
from pytest_mock import MockerFixture

from mlip_autopipec.config_schemas import (
    DFTConfig,
    DFTInput,
    ExplorerParams,
    FPSParams,
    SOAPParams,
    SurrogateModelParams,
    SystemConfig,
)
from mlip_autopipec.modules.explorer import SurrogateExplorer


@pytest.fixture
def mock_system_config() -> SystemConfig:
    """Create a mock SystemConfig for testing the explorer."""
    # This is a dummy config. We will override parts of it in specific tests.
    return SystemConfig(
        dft=DFTConfig(input=DFTInput(pseudopotentials={"H": "H.upf"})),
        explorer=ExplorerParams(
            surrogate_model=SurrogateModelParams(
                model_path="dummy/path/to/model.model", energy_threshold_ev=0.0
            ),
            fps=FPSParams(n_select=3, soap_params=SOAPParams()),
        ),
    )


def test_fps_selection_logic() -> None:
    """Unit test the Farthest Point Sampling algorithm's core logic."""
    # Create a simple, 2D dataset where the selection is obvious.
    # Three points are clustered, and three are far apart.
    descriptors = np.array(
        [
            [0, 0],
            [0.1, 0.1],
            [0.2, 0],
            [10, 10],
            [20, 0],
            [0, 20],
        ]
    )
    n_select = 3

    # Manually instantiate a dummy explorer to access the private method.
    # We don't need a full config for this unit test.
    dummy_surrogate_params = SurrogateModelParams(
        model_path="dummy", energy_threshold_ev=0.0
    )
    dummy_explorer_config = ExplorerParams(surrogate_model=dummy_surrogate_params)
    explorer = SurrogateExplorer(
        SystemConfig(
            dft=DFTConfig(input=DFTInput(pseudopotentials={})),
            explorer=dummy_explorer_config,
        )
    )

    # Fix the random seed for predictable starting point
    np.random.seed(0)
    selected_indices = explorer._farthest_point_sampling(descriptors, n_select)

    # With seed(0), the start is index 4 ([20, 0]).
    # The next farthest is index 5 ([0, 20]).
    # The next farthest is index 0 ([0, 0]), because it is far from both 4 and 5.
    assert sorted(selected_indices) == [0, 4, 5]


def test_select_pipeline(
    mocker: MockerFixture, mock_system_config: SystemConfig
) -> None:
    """Integration test for the full select pipeline using mocks."""
    # 1. Test Fixture: Create a list of 10 simple Atoms objects.
    candidates = [Atoms("H", positions=[(0, 0, i)]) for i in range(10)]

    # 2. Configuration
    assert mock_system_config.explorer is not None
    mock_system_config.explorer.surrogate_model.energy_threshold_ev = -5.0
    mock_system_config.explorer.fps.n_select = 3

    # 3. Mock Behavior
    # Mock the MACE model to return high energies for 3 atoms.
    mock_calculator = mocker.patch("mlip_autopipec.modules.explorer.mace_mp")
    energies = [-10.0] * 7 + [1.0] * 3  # 7 pass, 3 fail
    mock_calculator.return_value.get_potential_energy.side_effect = energies

    # Mock dscribe to return pre-defined descriptors for the 7 survivors.
    mock_soap = mocker.patch("mlip_autopipec.modules.explorer.SOAP")
    # Make the first three descriptors the most diverse
    descriptors = np.array(
        [
            [10, 10],
            [20, 0],
            [0, 20],
            [0, 0],
            [0.1, 0.1],
            [0.2, 0],
            [-0.1, 0.1],
        ]
    )
    mock_soap.return_value.create.return_value = descriptors

    # 4. Execution
    np.random.seed(42)  # Ensure reproducibility for FPS
    explorer = SurrogateExplorer(mock_system_config)
    final_selection = explorer.select(candidates)

    # 5. Assertions
    # Assert that the final list has the correct size.
    assert len(final_selection) == 3

    # Assert that the high-energy structures were filtered out.
    # The final selection should be a subset of the first 7 candidates.
    final_indices = {c.positions[0, 2] for c in final_selection}
    assert final_indices.issubset(set(range(7)))

    # Assert that FPS selected the most diverse structures.
    # With seed(42), start is index 1. Next are 2 and 6.
    assert final_indices == {1, 2, 6}
