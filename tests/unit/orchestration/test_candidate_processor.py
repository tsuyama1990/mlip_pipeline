from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from mlip_autopipec.domain_models.candidate import CandidateConfig
from mlip_autopipec.orchestration.candidate import CandidateProcessor


@pytest.fixture
def mock_config():
    return CandidateConfig(
        perturbation_radius=0.1,
        num_perturbations=2,
        cluster_cutoff=5.0
    )


@pytest.fixture
def mock_pacemaker():
    pm = MagicMock()
    # Mock select_active_set to return a subset of indices
    pm.select_active_set.return_value = [0, 2]
    return pm


@pytest.fixture
def sample_atoms():
    return Atoms("H", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)


def test_candidate_processor_process(mock_config, mock_pacemaker, sample_atoms, tmp_path):
    # Setup inputs
    halted_structure_path = tmp_path / "halted.dump"
    halted_structure_path.touch()
    # Create a dummy file (CandidateProcessor should mock read, but let's assume it reads this path)

    potential_path = tmp_path / "potential.yace"
    potential_path.touch()

    processor = CandidateProcessor(mock_config, mock_pacemaker)

    # Mock ase.io.read to return our sample atoms
    with patch("mlip_autopipec.orchestration.candidate.read") as mock_read:
        # Return a copy to avoid side effects across calls
        mock_read.side_effect = lambda *args, **kwargs: sample_atoms.copy()

        # We also need to mock ClusterEmbedder inside CandidateProcessor
        with patch("mlip_autopipec.orchestration.candidate.ClusterEmbedder") as MockEmbedder:
            embedder_instance = MockEmbedder.return_value
            # return copies of atoms as "embedded"
            embedder_instance.embed.side_effect = lambda atoms, **kwargs: atoms.copy()

            # Execute
            candidates = processor.process(halted_structure_path, potential_path, elements=["Al"])

            # Assertions
            mock_pacemaker.select_active_set.assert_called_once()
            mock_pacemaker.reset_mock()

            # We generated 2 perturbations.
            # select_active_set returns indices [0, 2] -> wait, indices of what?
            # If we generate 2 perturbations + 1 original = 3 candidates?
            # Let's assume processor logic generates N perturbations.
            # If it generates `num_perturbations` (2), then total pool is 2.
            # If select_active_set returns [0], then we get 1 candidate.

            assert isinstance(candidates, list)
            # mock_pacemaker returned [0, 2], but if we only had 2 perturbations, index 2 is out of bounds?
            # We need to align mock return with logic.
            # If num_perturbations=2, we have 2 candidates. Indices 0 and 1.
            # Let's update mock_pacemaker to return [0]
            mock_pacemaker.select_active_set.return_value = [0]

            candidates = processor.process(halted_structure_path, potential_path, elements=["Al"])

            assert len(candidates) == 1
            assert isinstance(candidates[0], Atoms)

            # Check interaction with Pacemaker
            mock_pacemaker.select_active_set.assert_called_once()
            args, _ = mock_pacemaker.select_active_set.call_args
            assert args[1] == potential_path # Second arg is potential
