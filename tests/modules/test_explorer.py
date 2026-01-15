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
    mocker.patch(
        "mlip_autopipec.modules.explorer.mace_mp"
    ).return_value.get_potential_energy.side_effect = [-10.0] * 7 + [1.0] * 3

    # Mock dscribe to return pre-defined descriptors for the 7 survivors.
    mocker.patch(
        "mlip_autopipec.modules.descriptors.SOAPDescriptorCalculator.calculate"
    ).return_value = np.random.rand(7, 10)

    # 4. Execution
    np.random.seed(42)  # Ensure reproducibility for FPS
    explorer = SurrogateExplorer(mock_system_config)
    final_selection = explorer.select(candidates)

    # 5. Assertions
    assert len(final_selection) == 3
    final_indices = {c.positions[0, 2] for c in final_selection}
    assert final_indices.issubset(set(range(7)))


def test_empty_candidate_list(mock_system_config: SystemConfig) -> None:
    """Test that an empty candidate list is handled gracefully."""
    explorer = SurrogateExplorer(mock_system_config)
    final_selection = explorer.select([])
    assert final_selection == []


def test_all_candidates_screened(
    mocker: MockerFixture, mock_system_config: SystemConfig
) -> None:
    """Test that if all candidates are screened out, an empty list is returned."""
    candidates = [Atoms("H")] * 5
    assert mock_system_config.explorer is not None
    mock_system_config.explorer.surrogate_model.energy_threshold_ev = -5.0

    mocker.patch(
        "mlip_autopipec.modules.explorer.mace_mp"
    ).return_value.get_potential_energy.return_value = 1.0  # All are high-energy

    explorer = SurrogateExplorer(mock_system_config)
    final_selection = explorer.select(candidates)
    assert final_selection == []


def test_fewer_candidates_than_n_select(
    mocker: MockerFixture, mock_system_config: SystemConfig
) -> None:
    """Test that if fewer candidates remain than n_select, all are returned."""
    candidates = [Atoms("H")] * 5
    assert mock_system_config.explorer is not None
    mock_system_config.explorer.surrogate_model.energy_threshold_ev = -5.0
    mock_system_config.explorer.fps.n_select = 10

    mocker.patch(
        "mlip_autopipec.modules.explorer.mace_mp"
    ).return_value.get_potential_energy.return_value = -10.0  # All pass

    explorer = SurrogateExplorer(mock_system_config)
    final_selection = explorer.select(candidates)
    assert len(final_selection) == 5


def test_invalid_energy_values(
    mocker: MockerFixture, mock_system_config: SystemConfig
) -> None:
    """Test that NaN and Inf energy values are handled correctly."""
    candidates = [Atoms("H")] * 3
    assert mock_system_config.explorer is not None
    mock_system_config.explorer.surrogate_model.energy_threshold_ev = -5.0

    mocker.patch(
        "mlip_autopipec.modules.explorer.mace_mp"
    ).return_value.get_potential_energy.side_effect = [
        -10.0,
        np.nan,
        np.inf,
    ]  # One valid, two invalid

    explorer = SurrogateExplorer(mock_system_config)
    final_selection = explorer.select(candidates)
    assert len(final_selection) == 1
