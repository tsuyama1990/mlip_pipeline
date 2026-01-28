from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from mlip_autopipec.surrogate.candidate_manager import CandidateManager


@pytest.fixture
def mock_db():
    return MagicMock()

@pytest.fixture
def mock_pacemaker():
    return MagicMock()

@pytest.fixture
def manager(mock_db, mock_pacemaker):
    return CandidateManager(mock_db, mock_pacemaker)

def test_process_halted_flow(manager, mock_pacemaker, tmp_path):
    """
    Verifies the flow: Extract -> Perturb -> Select -> Embed -> Save
    """
    dump_path = tmp_path / "fake.dump"
    dump_path.touch()
    potential_path = Path("fake.yace")

    # Create a dummy atom
    atoms = Atoms("Si", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)

    # Mock internal methods
    # We patch the methods on the instance or class. Since we haven't implemented them yet,
    # we'll assume they exist or will exist.

    with patch.object(manager, "_extract_cluster", return_value=atoms) as mock_extract, \
         patch.object(manager, "_perturb_structure", return_value=[atoms, atoms]) as mock_perturb, \
         patch.object(manager, "_embed_structure", return_value=atoms) as mock_embed, \
         patch("mlip_autopipec.surrogate.candidate_manager.write"), \
         patch("mlip_autopipec.surrogate.candidate_manager.tempfile.NamedTemporaryFile") as mock_tmp:

        # Mock select_active_set to return index 0
        mock_pacemaker.select_active_set.return_value = [0]

        # Mock tmp file context manager
        mock_tmp.return_value.__enter__.return_value.name = str(tmp_path / "candidates.xyz")

        manager.process_halted(dump_path, potential_path, n_perturbations=2)

        # Verification
        mock_extract.assert_called_with(dump_path)
        mock_perturb.assert_called_once()

        # Verify selection was called
        mock_pacemaker.select_active_set.assert_called()

        # Only the selected candidate (index 0) should be embedded and saved
        assert mock_embed.call_count == 1
        manager.db.add_structure.assert_called_once() # Assuming add_structure is used
