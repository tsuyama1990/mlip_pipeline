"""
Unit tests for the DFTHeuristics class.
"""
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from ase import Atoms
from ase.build import bulk, molecule

from mlip_autopipec.modules.dft import DFTHeuristics


@pytest.fixture
def mock_sssp_data(tmp_path: Path) -> Path:
    """Creates a mock SSSP data file for testing."""
    sssp_data = {
        "Si": {"cutoff_wfc": 60, "cutoff_rho": 240, "filename": "Si.upf"},
        "Fe": {"cutoff_wfc": 80, "cutoff_rho": 320, "filename": "Fe.upf"},
        "O": {"cutoff_wfc": 70, "cutoff_rho": 280, "filename": "O.upf"},
        "H": {"cutoff_wfc": 50, "cutoff_rho": 200, "filename": "H.upf"},
    }
    sssp_path = tmp_path / "sssp.json"
    with sssp_path.open("w") as f:
        json.dump(sssp_data, f)
    return sssp_path


def test_dft_heuristics_cutoffs(mock_sssp_data: Path):
    """Tests that the cutoffs are correctly determined from the SSSP data."""
    heuristics = DFTHeuristics(sssp_data_path=mock_sssp_data)
    atoms = bulk("Si")
    params = heuristics.get_heuristic_parameters(atoms)
    assert params.cutoffs.wavefunction == 60
    assert params.cutoffs.density == 240


def test_dft_heuristics_k_points(mock_sssp_data: Path):
    """Tests the k-point generation for a bulk system."""
    heuristics = DFTHeuristics(sssp_data_path=mock_sssp_data)
    atoms = bulk("Si", a=5.43)
    params = heuristics.get_heuristic_parameters(atoms)
    # Expected: max(1, int(6.0 / 5.43) + 1) = 2
    assert params.k_points == (2, 2, 2)


def test_dft_heuristics_k_points_2d(mock_sssp_data: Path):
    """Tests the k-point generation for a 2D system (slab)."""
    heuristics = DFTHeuristics(sssp_data_path=mock_sssp_data)
    atoms = bulk("Si", a=5.43)
    atoms.cell[2] = [0, 0, 20]
    params = heuristics.get_heuristic_parameters(atoms)
    assert params.k_points == (2, 2, 1)


def test_dft_heuristics_magnetism(mock_sssp_data: Path):
    """Tests that magnetism is enabled for magnetic elements."""
    heuristics = DFTHeuristics(sssp_data_path=mock_sssp_data)
    atoms = bulk("Fe")
    params = heuristics.get_heuristic_parameters(atoms)
    assert params.magnetism is not None
    assert params.magnetism.nspin == 2
    assert params.magnetism.starting_magnetization.root["Fe"] == 0.5


def test_dft_heuristics_no_magnetism(mock_sssp_data: Path):
    """Tests that magnetism is disabled for non-magnetic elements."""
    heuristics = DFTHeuristics(sssp_data_path=mock_sssp_data)
    atoms = bulk("Si")
    params = heuristics.get_heuristic_parameters(atoms)
    assert params.magnetism is None


def test_dft_heuristics_pseudopotentials(mock_sssp_data: Path):
    """Tests that the correct pseudopotentials are selected."""
    heuristics = DFTHeuristics(sssp_data_path=mock_sssp_data)
    atoms = molecule("H2O")
    params = heuristics.get_heuristic_parameters(atoms)
    assert params.pseudopotentials.root["H"] == "H.upf"
    assert params.pseudopotentials.root["O"] == "O.upf"


def test_dft_heuristics_missing_element(tmp_path: Path):
    """Tests that an error is raised if an element is not in the SSSP data."""
    sssp_data = {"Si": {"cutoff_wfc": 60, "cutoff_rho": 240, "filename": "Si.upf"}}
    sssp_path = tmp_path / "sssp.json"
    with sssp_path.open("w") as f:
        json.dump(sssp_data, f)
    heuristics = DFTHeuristics(sssp_data_path=sssp_path)
    atoms = bulk("Ge")  # Germanium is not in our mock data
    with pytest.raises(KeyError):
        heuristics.get_heuristic_parameters(atoms)
