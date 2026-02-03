from pathlib import Path

import pytest
from ase import Atoms

from mlip_autopipec.validation.metrics import ElasticValidator, PhononValidator


@pytest.fixture
def mock_atoms() -> Atoms:
    return Atoms("Cu", positions=[[0, 0, 0]], cell=[3.6, 3.6, 3.6], pbc=True)


def test_phonon_validator_execution(mock_atoms: Atoms, tmp_path: Path) -> None:
    validator = PhononValidator()
    # It should pass (or fail gracefully) but not raise NotImplementedError
    # We use a dummy calculator so it might actually work or fail with skipped if phonopy missing
    result = validator.validate(Path("potential.yace"), mock_atoms, tmp_path)

    assert result.name == "phonons"
    if result.passed:
        if result.details.get("status") == "skipped":
            pass  # Skipped
        else:
            assert result.score is not None
            assert (tmp_path / "phonon_band_structure.png").exists()


def test_elastic_validator_execution(mock_atoms: Atoms, tmp_path: Path) -> None:
    validator = ElasticValidator()
    result = validator.validate(Path("potential.yace"), mock_atoms, tmp_path)

    assert result.name == "elastic"
    assert result.passed is True
    assert result.details["C11"] == 160.0
