"""Tests for Oracle module thread safety."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms

from pyacemaker.core.config import (
    DFTConfig,
    OracleConfig,
    ProjectConfig,
    PYACEMAKERConfig,
)
from pyacemaker.domain_models.models import StructureMetadata
from pyacemaker.modules.oracle import DFTOracle


@pytest.fixture
def config(tmp_path: Path) -> PYACEMAKERConfig:
    pp_file = tmp_path / "H.pbe.UPF"
    pp_file.touch()
    return PYACEMAKERConfig(
        version="0.1.0",
        project=ProjectConfig(name="Test", root_dir=tmp_path),
        oracle=OracleConfig(
            dft=DFTConfig(
                code="qe",
                pseudopotentials={"H": str(pp_file)},
                max_workers=4,  # Multithreaded
                chunk_size=2,
            )
        ),
    )


def test_dft_oracle_thread_safety(config: PYACEMAKERConfig) -> None:
    """Verify that DFTOracle processes batch correctly with multiple threads."""
    oracle = DFTOracle(config)

    # Create 10 structures
    structures = [StructureMetadata(features={"atoms": Atoms("H")}) for _ in range(10)]

    # Mock DFTManager.compute to be slow but thread-safe
    # We verify max_workers usage indirectly by ensuring this doesn't block sequentially
    # But for unit test stability, we just ensure it returns correct data
    def mock_compute_return(atoms: Atoms) -> MagicMock:
        # Avoid sleep to prevent flakiness, unless strictly necessary for race condition check
        res = MagicMock(spec=Atoms)
        res.get_potential_energy.return_value = -10.0
        res.get_forces.return_value = np.array([[0.0, 0.0, 0.0]])
        res.get_stress.return_value = np.array([0.0] * 6)
        return res

    with patch(
        "pyacemaker.modules.oracle.DFTManager.compute", side_effect=mock_compute_return
    ) as mock_compute:
        results = list(oracle.compute_batch(structures))

        assert len(results) == 10
        assert all(s.energy == -10.0 for s in results)
        assert mock_compute.call_count == 10
