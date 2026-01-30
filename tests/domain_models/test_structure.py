
import ase
import numpy as np
import pytest
from pathlib import Path
from pydantic import ValidationError

from mlip_autopipec.domain_models.structure import Structure, JobStatus, JobResult, LammpsResult


def test_structure_valid() -> None:
    """Test creating a valid structure."""
    s = Structure(
        symbols=["H", "H"],
        positions=np.array([[0, 0, 0], [0, 0, 0.74]]),
        cell=np.eye(3) * 10,
        pbc=(True, True, True),
        properties={"test": 1}
    )
    assert s.symbols == ["H", "H"]
    assert s.positions.shape == (2, 3)

def test_structure_consistency_error() -> None:
    """Test mismatch between symbols and positions."""
    with pytest.raises(ValidationError) as excinfo:
        Structure(
            symbols=["H", "H"],
            positions=np.array([[0, 0, 0], [0, 0, 0.74], [0, 0, 1.0]]), # 3 positions
            cell=np.eye(3) * 10,
            pbc=(True, True, True)
        )
    assert "Number of symbols (2) does not match number of positions (3)" in str(excinfo.value)

def test_structure_cell_shape_error() -> None:
    """Test invalid cell shape."""
    with pytest.raises(ValidationError) as excinfo:
        Structure(
            symbols=["H"],
            positions=np.array([[0, 0, 0]]),
            cell=np.eye(2) * 10, # 2x2 cell
            pbc=(True, True, True)
        )
    assert "Cell must be a 3x3 matrix" in str(excinfo.value)

def test_structure_positions_shape_error() -> None:
    """Test invalid positions shape."""
    with pytest.raises(ValidationError) as excinfo:
        Structure(
            symbols=["H"],
            positions=np.array([0, 0, 0]), # 1D array
            cell=np.eye(3) * 10,
            pbc=(True, True, True)
        )
    assert "Positions must have shape" in str(excinfo.value)

def test_from_ase_to_ase_roundtrip(sample_ase_atoms: ase.Atoms) -> None:
    """Test ASE conversion roundtrip."""
    s = Structure.from_ase(sample_ase_atoms)
    assert s.symbols == sample_ase_atoms.get_chemical_symbols()  # type: ignore[no-untyped-call]
    np.testing.assert_array_equal(s.positions, sample_ase_atoms.get_positions())  # type: ignore[no-untyped-call]
    np.testing.assert_array_equal(s.cell, sample_ase_atoms.get_cell().array)  # type: ignore[no-untyped-call]
    assert s.pbc == tuple(sample_ase_atoms.get_pbc())  # type: ignore[no-untyped-call]
    assert s.properties == sample_ase_atoms.info

    atoms_back = s.to_ase()
    assert atoms_back.get_chemical_formula() == sample_ase_atoms.get_chemical_formula()  # type: ignore[no-untyped-call]
    np.testing.assert_array_almost_equal(atoms_back.get_positions(), sample_ase_atoms.get_positions())  # type: ignore[no-untyped-call]
    assert atoms_back.info == sample_ase_atoms.info

def test_job_status_enum() -> None:
    """Test JobStatus enum."""
    assert JobStatus.PENDING == "PENDING"
    assert JobStatus.TIMEOUT == "TIMEOUT"

def test_job_result_valid() -> None:
    """Test creating a valid JobResult."""
    jr = JobResult(
        job_id="123",
        status=JobStatus.COMPLETED,
        work_dir=Path("/tmp"),
        duration_seconds=10.5
    )
    assert jr.status == JobStatus.COMPLETED

def test_lammps_result_valid() -> None:
    """Test creating a valid LammpsResult with structure."""
    s = Structure(
        symbols=["H"],
        positions=np.array([[0, 0, 0]]),
        cell=np.eye(3),
        pbc=(True, True, True)
    )

    lr = LammpsResult(
        job_id="456",
        status=JobStatus.COMPLETED,
        work_dir=Path("/tmp"),
        duration_seconds=5.0,
        final_structure=s,
        trajectory_path=Path("traj.dump")
    )
    assert lr.final_structure is not None
    assert lr.final_structure.symbols == ["H"]

def test_lammps_result_failed() -> None:
    """Test creating a LammpsResult for a failed job."""
    lr = LammpsResult(
        job_id="789",
        status=JobStatus.FAILED,
        work_dir=Path("/tmp"),
        duration_seconds=0.1
    )
    assert lr.final_structure is None
