import numpy as np
import pytest
from ase import Atoms
from pydantic import ValidationError
from pathlib import Path

from mlip_autopipec.domain_models.inputs import Job, ProjectState, Structure, sanitize_value


def test_sanitize_value() -> None:
    """Test recursive sanitization of numpy types."""
    # Test numpy scalars
    assert isinstance(sanitize_value(np.int64(10)), int)
    assert isinstance(sanitize_value(np.float64(10.5)), float)
    assert isinstance(sanitize_value(np.bool_(True)), bool)

    # Test numpy arrays
    arr = np.array([1, 2, 3])
    sanitized = sanitize_value(arr)
    assert isinstance(sanitized, list)
    assert sanitized == [1, 2, 3]

    # Test complex structure (dict with mixed types)
    data = {
        "scalar": np.int64(5),
        "array": np.array([1.0, 2.0]),
        "nested": {
            "val": np.bool_(False)
        },
        "list": [np.int32(1), np.float32(2.0)]
    }
    cleaned = sanitize_value(data)
    assert isinstance(cleaned["scalar"], int)
    assert isinstance(cleaned["array"], list)
    assert isinstance(cleaned["nested"]["val"], bool)
    assert isinstance(cleaned["list"][0], int)


def test_structure_ase_conversion() -> None:
    """Test conversion between ASE Atoms and Structure model."""
    atoms = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    atoms.info["energy"] = np.float64(-15.5)
    atoms.info["relaxed"] = np.bool_(True)

    # Convert to Structure
    struct = Structure.from_ase(atoms)

    assert struct.numbers == [1, 1, 8]
    assert isinstance(struct.tags["energy"], float)
    assert isinstance(struct.tags["relaxed"], bool)

    # Convert back to ASE
    atoms2 = struct.to_ase()
    assert np.allclose(atoms.positions, atoms2.positions)
    assert atoms.get_chemical_formula() == atoms2.get_chemical_formula() # type: ignore[no-untyped-call]
    assert atoms2.info["energy"] == -15.5


def test_structure_validation() -> None:
    """Test that invalid structures raise ValidationError."""
    # Missing required fields
    with pytest.raises(ValidationError):
        Structure(positions=[[0, 0, 0]]) # type: ignore[call-arg]

    # Wrong type
    with pytest.raises(ValidationError):
        Structure(
            positions="invalid", # type: ignore[arg-type]
            numbers=[1],
            cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            pbc=[True, True, True]
        )


def test_job_model(tmp_path: Path) -> None:
    """Test Job model creation."""
    job = Job(
        id="job_123",
        name="test_job",
        work_dir=tmp_path / "test",
        command="echo hello"
    )
    assert job.id == "job_123"
    assert job.status == "pending"


def test_project_state() -> None:
    """Test ProjectState model."""
    state = ProjectState(current_iteration=5)
    assert state.current_iteration == 5

    with pytest.raises(ValidationError):
        ProjectState(current_iteration=-1)
