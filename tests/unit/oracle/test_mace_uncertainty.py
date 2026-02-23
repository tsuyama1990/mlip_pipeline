"""Tests for MACE Surrogate Oracle Uncertainty."""

from pathlib import Path
from unittest.mock import patch

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
from pyacemaker.modules.oracle import MaceSurrogateOracle


@pytest.fixture
def mock_config(tmp_path: Path) -> PYACEMAKERConfig:
    """Mock configuration."""
    project_config = ProjectConfig(name="test", root_dir=tmp_path)
    # Configure MACE with mock=True
    oracle_config = OracleConfig(
        dft=DFTConfig(pseudopotentials={"Fe": "Fe.pbe"}),
        mace=MaceConfig(model_path="medium", mock=True),
        mock=True # Orchestrator uses this to select class, but MaceSurrogateOracle checks mace config
    )

    return PYACEMAKERConfig(
        version=CONSTANTS.default_version,
        project=project_config,
        oracle=oracle_config,
    )


def test_compute_uncertainty_mock(mock_config: PYACEMAKERConfig) -> None:
    """Test compute_uncertainty with mock configuration."""
    oracle = MaceSurrogateOracle(mock_config)

    # Create structures
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    structures = [
        StructureMetadata(features={"atoms": atoms.copy()}, status=StructureStatus.NEW) # type: ignore[no-untyped-call]
        for _ in range(5)
    ]

    # Compute
    results = list(oracle.compute_uncertainty(structures))

    assert len(results) == 5
    for s in results:
        assert s.uncertainty_state is not None
        # Mock returns 0.5
        assert s.uncertainty_state.gamma_mean == 0.5
        assert s.uncertainty_state.gamma_max == 0.5


def test_compute_uncertainty_validation_call(mock_config: PYACEMAKERConfig) -> None:
    """Verify validate_structure_integrity is called."""
    oracle = MaceSurrogateOracle(mock_config)

    # Structure with invalid atoms (e.g. no positions) - hard to construct passing Pydantic validation
    # But validate_structure_integrity checks for NaN etc.
    # Let's mock validate_structure_integrity

    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    structures = [StructureMetadata(features={"atoms": atoms}, status=StructureStatus.NEW)]

    with patch("pyacemaker.modules.oracle.validate_structure_integrity") as mock_validate:
        list(oracle.compute_uncertainty(structures))
        mock_validate.assert_called()


def test_compute_batch_validation_call(mock_config: PYACEMAKERConfig) -> None:
    """Verify validate_structure_integrity is called in compute_batch."""
    oracle = MaceSurrogateOracle(mock_config)

    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    structures = [StructureMetadata(features={"atoms": atoms}, status=StructureStatus.NEW)]

    with patch("pyacemaker.modules.oracle.validate_structure_integrity") as mock_validate:
        list(oracle.compute_batch(structures))
        mock_validate.assert_called()


def test_batch_processing_optimization(mock_config: PYACEMAKERConfig) -> None:
    """Verify batch processing uses internal chunking."""
    # Let's switch off mock mode in config but mock MaceManager
    mock_config.oracle.mock = False
    assert mock_config.oracle.mace is not None
    mock_config.oracle.mace.mock = False

    with patch("pyacemaker.modules.oracle.MaceManager") as MockManager:
        mock_instance = MockManager.return_value
        # Return uncertainty for batch
        mock_instance.compute_uncertainty.side_effect = lambda atoms_list: [0.1] * len(atoms_list)

        oracle = MaceSurrogateOracle(mock_config)

        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
        # 150 structures, default chunk is 100
        structures = [
            StructureMetadata(features={"atoms": atoms.copy()}, status=StructureStatus.NEW) # type: ignore[no-untyped-call]
            for _ in range(150)
        ]

        results = list(oracle.compute_uncertainty(structures))

        # Should call compute_uncertainty twice (100 + 50)
        assert mock_instance.compute_uncertainty.call_count == 2
        assert len(results) == 150


def test_mace_compute_batch_mock(mock_config: PYACEMAKERConfig) -> None:
    """Test compute_batch with mock configuration."""
    oracle = MaceSurrogateOracle(mock_config)

    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    structures = [
        StructureMetadata(features={"atoms": atoms.copy()}, status=StructureStatus.NEW) # type: ignore[no-untyped-call]
        for _ in range(5)
    ]

    results = list(oracle.compute_batch(structures))

    assert len(results) == 5
    for s in results:
        assert s.status == StructureStatus.CALCULATED
        assert s.energy is not None
        assert s.forces is not None
        assert s.label_source == "mace"
