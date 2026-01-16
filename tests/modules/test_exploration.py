"""
Unit and integration tests for the SurrogateExplorer module.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from mlip_autopipec.config.models import ExplorerConfig, FingerprintConfig
from mlip_autopipec.modules.exploration import SurrogateExplorer


@pytest.fixture
def mock_explorer_config(tmp_path) -> ExplorerConfig:
    """Provides a mock ExplorerConfig for testing."""
    # Create a dummy model file
    model_path = tmp_path / "mock_model.pt"
    model_path.touch()

    fingerprint_config = FingerprintConfig(
        type="soap",
        species=["Si"],
        soap_rcut=4.0,
        soap_nmax=6,
        soap_lmax=4,
    )
    return ExplorerConfig(
        surrogate_model_path=str(model_path),
        max_force_threshold=5.0,
        fingerprint=fingerprint_config,
    )


# --- Unit Tests ---


def test_fps_selects_outliers_and_diverse_points():
    """
    Tests the Farthest Point Sampling (FPS) algorithm with a deterministic
    2D dataset to ensure it prioritizes outliers and diverse points.
    """
    # Create a clustered set of points and a few distinct outliers
    outliers = np.array([[1, 1], [-1, -1], [1, -1]])
    cluster = np.random.rand(20, 2) * 0.1
    points = np.vstack([outliers, cluster])

    selected_indices = SurrogateExplorer._farthest_point_sampling(points, 3)

    # Assert that all three outliers were selected by checking their indices
    outlier_indices = {0, 1, 2}
    assert outlier_indices.issubset(set(selected_indices))


def test_fps_returns_correct_number_of_points():
    """Ensures FPS returns the requested number of indices."""
    points = np.random.rand(50, 3)
    selected_indices = SurrogateExplorer._farthest_point_sampling(points, 10)
    assert len(selected_indices) == 10


def test_fps_handles_fewer_points_than_requested():
    """Ensures FPS returns all points if fewer are available than requested."""
    points = np.random.rand(5, 3)
    selected_indices = SurrogateExplorer._farthest_point_sampling(points, 10)
    assert len(selected_indices) == 5
    assert set(selected_indices) == set(range(5))


def test_fingerprint_calculation_is_deterministic(mocker, mock_explorer_config):
    """
    Validates that the fingerprint generation is deterministic by running it
    twice on the same structure and asserting the outputs are identical.
    """
    mocker.patch(
        "mlip_autopipec.modules.exploration.mace_mp", return_value=mocker.MagicMock()
    )

    mock_soap_create = mocker.patch(
        "dscribe.descriptors.SOAP.create",
        return_value=np.array([[0.1, 0.2, 0.3]]),
    )

    structure = bulk("Si")
    explorer = SurrogateExplorer(mock_explorer_config)
    fingerprint1 = explorer._calculate_fingerprints([structure])
    fingerprint2 = explorer._calculate_fingerprints([structure])

    assert np.array_equal(fingerprint1, fingerprint2)
    assert mock_soap_create.call_count == 2


def test_pre_screening_filters_unstable_structures(mocker, mock_explorer_config):
    """
    Tests that the pre-screening logic correctly identifies and removes
    structures with forces exceeding the defined threshold.
    """
    mock_mace_calc = mocker.MagicMock()

    stable_forces = np.array([[0.1, 0.0, -0.1], [-0.1, 0.0, 0.1]])
    unstable_forces = np.array([[10.0, 5.0, 0.0], [-10.0, -5.0, 0.0]])

    def get_forces_side_effect(atoms):
        if atoms.info.get("id") == "unstable":
            return unstable_forces
        return stable_forces

    # Correctly patch mace_mp where it's looked up: in the exploration module
    mock_mace_fn = mocker.patch(
        "mlip_autopipec.modules.exploration.mace_mp", return_value=mock_mace_calc
    )
    mock_mace_calc.get_forces.side_effect = get_forces_side_effect

    stable_structure = Atoms("Si2", positions=[[0, 0, 0], [1, 1, 1]], info={"id": "stable"})
    unstable_structure = Atoms(
        "Si2", positions=[[0, 0, 0], [0.1, 0.1, 0.1]], info={"id": "unstable"}
    )

    explorer = SurrogateExplorer(mock_explorer_config)
    screened_structures = explorer._pre_screen_structures(
        [stable_structure, unstable_structure]
    )

    assert len(screened_structures) == 1
    assert screened_structures[0].info.get("id") == "stable"
    mock_mace_fn.assert_called_once()


# --- Integration Tests Skeleton ---


@pytest.mark.integration
def test_select_pipeline_end_to_end(mock_explorer_config):
    """
    **Integration Test Skeleton**
    Tests the full `select` pipeline: pre-screening -> fingerprinting -> FPS.
    This test will use a real (but small/fast) MACE model and dscribe.
    """
    pytest.skip("Integration test not yet implemented.")


@pytest.mark.integration
def test_pipeline_with_no_survivors(mock_explorer_config):
    """
    **Integration Test Skeleton**
    Tests the edge case where pre-screening eliminates all candidate structures.
    """
    pytest.skip("Integration test not yet implemented.")
