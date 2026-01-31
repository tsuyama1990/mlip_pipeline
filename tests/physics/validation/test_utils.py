import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from mlip_autopipec.domain_models.config import ValidationConfig, PotentialConfig
from mlip_autopipec.physics.validation.utils import get_reference_structure

def test_get_reference_structure_valid():
    val_config = ValidationConfig(
        ref_crystal_structure="fcc",
        ref_lattice_constant=4.0
    )
    pot_config = PotentialConfig(elements=["Al"], cutoff=5.0)

    with patch("ase.build.bulk") as mock_bulk:
        mock_atoms = MagicMock()
        # Setup mocks for Structure.from_ase
        mock_atoms.get_chemical_symbols.return_value = ["Al"]
        mock_atoms.get_positions.return_value = np.zeros((1,3))
        mock_atoms.get_cell.return_value.array = np.eye(3)
        mock_atoms.get_pbc.return_value = [True, True, True]
        mock_atoms.info.copy.return_value = {}

        mock_bulk.return_value = mock_atoms

        struct = get_reference_structure(val_config, pot_config)

        mock_bulk.assert_called_with("Al", "fcc", a=4.0)
        assert struct is not None

def test_get_reference_structure_missing_config():
    val_config = ValidationConfig(
        ref_crystal_structure=None,
        ref_lattice_constant=None
    )
    pot_config = PotentialConfig(elements=["Al"], cutoff=5.0)

    with pytest.raises(ValueError, match="ValidationConfig missing"):
        get_reference_structure(val_config, pot_config)

def test_get_reference_structure_ase_error():
    val_config = ValidationConfig(
        ref_crystal_structure="invalid",
        ref_lattice_constant=4.0
    )
    pot_config = PotentialConfig(elements=["Al"], cutoff=5.0)

    with patch("ase.build.bulk", side_effect=Exception("ASE Error")):
         with pytest.raises(ValueError, match="Could not generate reference structure"):
            get_reference_structure(val_config, pot_config)
