from pathlib import Path

import pytest
from ase import Atoms
from pydantic import ValidationError

from mlip_autopipec.domain_models.structure_enums import CandidateStatus
from mlip_autopipec.orchestration.database import DatabaseManager


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test.db"


def test_select_generator(db_path):
    atoms = Atoms("H")
    with DatabaseManager(db_path) as db:
        for i in range(5):
            db.add_structure(atoms, {"status": CandidateStatus.PENDING.value, "id_val": i})

        # Test iterator
        gen = db.select(status=CandidateStatus.PENDING.value)
        results = list(gen)
        assert len(results) == 5
        assert isinstance(results[0], Atoms)

        # Verify streaming nature (conceptually, by checking type)
        from collections.abc import Generator

        assert isinstance(db.select(), Generator)


def test_validate_path_safety_valid(tmp_path):
    from mlip_autopipec.utils.config_utils import validate_path_safety

    p = tmp_path / "config.yaml"
    p.touch()

    safe = validate_path_safety(p)
    assert safe == p.resolve()


def test_validate_path_safety_invalid():
    from mlip_autopipec.utils.config_utils import validate_path_safety

    # Path that doesn't exist but we want to check safety logic if it were to resolve to something weird
    # Currently it just resolves.
    # Check simple resolution
    p = Path("nonexistent.yaml")
    safe = validate_path_safety(p)
    assert safe == p.resolve()


def test_pseudo_dir_validation(tmp_path):
    from mlip_autopipec.config.schemas.dft import DFTConfig

    # Empty dir - should fail
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValidationError) as exc:
        DFTConfig(pseudopotential_dir=tmp_path / "empty", ecutwfc=30.0, kspacing=0.1)
    assert "No .UPF" in str(exc.value)

    # Dir with UPF - should pass
    (tmp_path / "valid").mkdir()
    (tmp_path / "valid" / "Fe.upf").touch()

    config = DFTConfig(pseudopotential_dir=tmp_path / "valid", ecutwfc=30.0, kspacing=0.1)
    assert config.pseudopotential_dir == tmp_path / "valid"
