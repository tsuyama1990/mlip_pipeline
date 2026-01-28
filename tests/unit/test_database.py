from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms

from mlip_autopipec.config.models import MinimalConfig, SystemConfig, TargetSystem
from mlip_autopipec.domain_models.dft_models import DFTResult
from mlip_autopipec.domain_models.structure_enums import CandidateStatus
from mlip_autopipec.exceptions import DatabaseError
from mlip_autopipec.orchestration.database import DatabaseManager


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test.db"


def test_database_init(db_path):
    with DatabaseManager(db_path) as db:
        assert db_path.exists()
        assert db.count() == 0


def test_add_structure(db_path):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    metadata = {"status": CandidateStatus.PENDING.value, "generation": 0}

    with DatabaseManager(db_path) as db:
        uid = db.add_structure(atoms, metadata)
        assert uid == 1
        assert db.count() == 1

        entries = list(db.select_entries())
        assert entries[0][0] == 1
        assert len(entries[0][1]) == 2
        assert entries[0][1].info["status"] == CandidateStatus.PENDING.value


def test_update_status(db_path):
    atoms = Atoms("H")
    with DatabaseManager(db_path) as db:
        uid = db.add_structure(atoms, {"status": CandidateStatus.PENDING.value})
        db.update_status(uid, CandidateStatus.TRAINING.value)

        entries = list(db.select_entries())
        assert entries[0][1].info["status"] == CandidateStatus.TRAINING.value


def test_validate_atoms_nan(db_path):
    atoms = Atoms("H", positions=[[float("nan"), 0, 0]])
    with DatabaseManager(db_path) as db:
        # DatabaseManager wraps errors in DatabaseError
        with pytest.raises(DatabaseError) as exc:
             db.add_structure(atoms, {})
        assert "NaN or Inf" in str(exc.value)


def test_validate_atoms_zero_cell_pbc(db_path):
    atoms = Atoms("H", cell=[0, 0, 0], pbc=True)
    with DatabaseManager(db_path) as db:
        with pytest.raises(DatabaseError) as exc:
            db.add_structure(atoms, {})
        assert "zero cell volume" in str(exc.value)


def test_count_kwargs(db_path):
    atoms = Atoms("H")
    with DatabaseManager(db_path) as db:
        db.add_structure(atoms, {"status": CandidateStatus.PENDING.value})
        db.add_structure(atoms, {"status": CandidateStatus.COMPLETED.value})

        assert db.count(selection=f"status={CandidateStatus.PENDING.value}") == 1
        assert db.count(selection=f"status={CandidateStatus.COMPLETED.value}") == 1


def test_update_metadata(db_path):
    atoms = Atoms("H")
    with DatabaseManager(db_path) as db:
        uid = db.add_structure(atoms, {"status": CandidateStatus.PENDING.value})
        db.update_metadata(uid, {"new_key": "value"})

        entries = list(db.select_entries())
        assert entries[0][1].info["new_key"] == "value"


def test_get_atoms(db_path):
    atoms = Atoms("H")
    with DatabaseManager(db_path) as db:
        db.add_structure(atoms, {"status": CandidateStatus.PENDING.value, "foo": "bar"})

        fetched = list(db.get_atoms(selection=f"status={CandidateStatus.PENDING.value}"))
        assert len(fetched) == 1
        assert fetched[0].info["foo"] == "bar"
        assert fetched[0].info["status"] == CandidateStatus.PENDING.value


def test_save_candidates(db_path):
    atoms = Atoms("H")
    with DatabaseManager(db_path) as db:
        db.save_candidates([atoms], cycle_index=1, method="random")
        assert db.count() == 1
        atoms_list = list(db.get_atoms())
        assert atoms_list[0].info["origin"] == "random"


def test_save_dft_result(db_path):
    atoms = Atoms("H")
    result = DFTResult(
        uid="test_uid",
        energy=-10.0,
        forces=[[0.0, 0.0, 0.0]],
        stress=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        succeeded=True,
        converged=True,
        wall_time=1.0,
        parameters={}
    )
    with DatabaseManager(db_path) as db:
        db.save_dft_result(atoms, result, {"status": CandidateStatus.COMPLETED.value})

        assert db.count() == 1
        saved = next(iter(db.get_atoms()))

        if saved.calc:
            assert saved.get_potential_energy() == -10.0
        else:
            assert saved.info.get("energy") == -10.0

        if saved.calc:
             np.testing.assert_allclose(saved.get_forces(), [[0.0, 0.0, 0.0]])
        else:
             np.testing.assert_allclose(saved.get_array("forces"), [[0.0, 0.0, 0.0]])

        assert saved.info["status"] == CandidateStatus.COMPLETED.value


def test_system_config(db_path):
    sys_conf = SystemConfig(
        target_system=TargetSystem(elements=["Fe"], composition={"Fe": 1.0}),
        minimal=MinimalConfig(target_system=TargetSystem(elements=["Fe"], composition={"Fe": 1.0})),
    )

    with DatabaseManager(db_path) as db:
        db.set_system_config(sys_conf)

    with DatabaseManager(db_path) as db:
        loaded = db.get_system_config()
        assert loaded.target_system.elements == ["Fe"]


# Error Handling Tests

def test_connect_error(db_path):
    with patch("mlip_autopipec.orchestration.database.connect", side_effect=Exception("Connection failed")):
        with pytest.raises(DatabaseError, match="Failed to initialize"):
             with DatabaseManager(db_path):
                 pass

def test_add_structure_error(db_path):
    atoms = Atoms("H")
    with DatabaseManager(db_path) as db:
        db._connection = MagicMock()
        db._connection.write.side_effect = Exception("Write failed")

        with pytest.raises(DatabaseError, match="Failed to add structure"):
            db.add_structure(atoms, {})

def test_update_status_error(db_path):
    with DatabaseManager(db_path) as db:
        db._connection = MagicMock()
        db._connection.update.side_effect = Exception("Update failed")
        with pytest.raises(DatabaseError, match="Failed to update status"):
            db.update_status(999, CandidateStatus.TRAINING.value)

def test_count_error(db_path):
    with DatabaseManager(db_path) as db:
        db._connection = MagicMock()
        db._connection.count.side_effect = Exception("Count failed")

        with pytest.raises(DatabaseError, match="Failed to count rows"):
            db.count()

def test_update_metadata_error(db_path):
    with DatabaseManager(db_path) as db:
        db._connection = MagicMock()
        db._connection.update.side_effect = Exception("Update failed")

        with pytest.raises(DatabaseError, match="Failed to update metadata"):
            db.update_metadata(1, {})

def test_save_dft_result_error(db_path):
    atoms = Atoms("H")
    result = DFTResult(
        uid="uid", energy=-10.0, forces=[[0.0,0.0,0.0]], succeeded=True, wall_time=0, parameters={}
    )

    with DatabaseManager(db_path) as db:
        db._connection = MagicMock()
        db._connection.write.side_effect = Exception("Write failed")

        with pytest.raises(DatabaseError, match="Failed to save DFT result"):
            db.save_dft_result(atoms, result, {})
