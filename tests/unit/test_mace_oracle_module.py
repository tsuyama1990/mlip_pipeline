"""Unit tests for MaceSurrogateOracle."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms

from pyacemaker.core.config import (
    CONSTANTS,
    DFTConfig,
    MaceConfig,
    OracleConfig,
    ProjectConfig,
    PYACEMAKERConfig,
)
from pyacemaker.domain_models.models import StructureMetadata, StructureStatus

# Bypass file checks
CONSTANTS.skip_file_checks = True


@pytest.fixture
def mock_config(tmp_path: Path) -> PYACEMAKERConfig:
    return PYACEMAKERConfig(
        version="0.1.0",
        project=ProjectConfig(name="test", root_dir=tmp_path),
        oracle=OracleConfig(
            dft=DFTConfig(pseudopotentials={"Fe": "Fe.pbe.UPF"}),
            mace=MaceConfig(model_path="medium"),
        ),
    )


@pytest.fixture
def mace_surrogate_oracle_class() -> type:
    from pyacemaker.modules.oracle import MaceSurrogateOracle

    return MaceSurrogateOracle


def test_init(
    mace_surrogate_oracle_class: type, mock_config: PYACEMAKERConfig
) -> None:
    oracle = mace_surrogate_oracle_class(mock_config)
    assert oracle.config == mock_config
    assert oracle.mace_manager is not None


@patch("pyacemaker.oracle.mace_oracle.MaceManager")
def test_compute_batch(
    mock_mace_manager: MagicMock,
    mace_surrogate_oracle_class: type,
    mock_config: PYACEMAKERConfig,
) -> None:
    # Setup
    # Create oracle instance. This will instantiate MaceManager(config.oracle.mace)
    oracle = mace_surrogate_oracle_class(mock_config)

    # MockMaceManager is the class, so calling it returns an instance
    # But wait, MaceSurrogateOracle.__init__ calls MaceManager(...)
    # If we patch the class, we get a mock instance.
    # However, creating oracle happens inside test function, so patch is active.
    # We need to grab the instance created inside __init__.

    # But wait, we are creating oracle instance *after* patch is applied decorator style?
    # Yes.

    # So oracle.mace_manager should be an instance of MockMaceManager.
    mock_manager = oracle.mace_manager

    # Mock manager.compute behavior
    def side_effect(atoms: Atoms) -> Atoms:
        calc_atoms = atoms.copy()  # type: ignore[no-untyped-call]
        if not isinstance(calc_atoms, Atoms):
            msg = "Expected Atoms object"
            raise TypeError(msg)
        # Mock calculator behavior on the atoms object
        # Since we use get_potential_energy() from ASE atoms, we mock the method on atoms object?
        # No, manager returns atoms with attached results usually?
        # In MaceManager.compute: it calls calc_structure.get_potential_energy() then returns calc_structure.
        # So it returns an atoms object that has calculator results cached?
        # Or simply returns atoms object.
        # BaseOracle._update_structure_common calls atoms.get_potential_energy().
        # So the returned atoms object must respond to get_potential_energy().

        # Mock the get_potential_energy method directly on the returned atoms object
        calc_atoms.get_potential_energy = MagicMock(return_value=-10.0)  # type: ignore[method-assign]
        calc_atoms.get_forces = MagicMock(  # type: ignore[method-assign]
            return_value=np.array([[0.0, 0.0, 0.0]] * len(atoms))
        )
        return calc_atoms

    # Mock manager.compute_batch behavior (since Oracle calls compute_batch)
    def batch_side_effect(atoms_list: list[Atoms]) -> list[Atoms]:
        return [side_effect(a) for a in atoms_list]

    mock_manager.compute_batch.side_effect = batch_side_effect

    # Input structures
    structures = [
        StructureMetadata(features={"atoms": Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])}),
        StructureMetadata(features={"atoms": Atoms("O", positions=[[0, 0, 0]])}),
    ]

    # Run
    results = list(oracle.compute_batch(structures))

    assert len(results) == 2
    assert results[0].status == StructureStatus.CALCULATED
    assert results[0].energy == -10.0
    # Check forces are present (mocked return value)
    assert results[0].forces is not None
    assert len(results[0].forces) == 2  # H2 has 2 atoms

    assert results[1].status == StructureStatus.CALCULATED
    assert results[1].energy == -10.0
    assert results[1].forces is not None
    assert len(results[1].forces) == 1  # O has 1 atom
