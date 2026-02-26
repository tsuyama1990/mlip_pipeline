"""Integration tests for error conditions."""

import gzip
import pickle
import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from pyacemaker.core.config import (
    DFTConfig,
    OracleConfig,
    OrchestratorConfig,
    ProjectConfig,
    PYACEMAKERConfig,
    StructureGeneratorConfig,
    TrainerConfig,
)
from pyacemaker.orchestrator import Orchestrator


class TestErrorConditions:
    """Tests for system behavior under error conditions."""

    @pytest.fixture
    def error_config(self, tmp_path: Path) -> PYACEMAKERConfig:
        """Create a config for error testing."""
        return PYACEMAKERConfig(
            version="0.1.0",
            project=ProjectConfig(name="ErrorTest", root_dir=tmp_path),
            orchestrator=OrchestratorConfig(max_cycles=1),
            structure_generator=StructureGeneratorConfig(strategy="random"),
            oracle=OracleConfig(
                dft=DFTConfig(code="qe", pseudopotentials={"H": "H.upf"}), mock=True
            ),
            trainer=TrainerConfig(mock=True),
        )

    def test_checksum_corruption_detection(self, error_config: PYACEMAKERConfig, tmp_path: Path) -> None:
        """Test that dataset loading fails if checksum is corrupted."""
        orchestrator = Orchestrator(error_config, base_dir=tmp_path)
        dataset_path = orchestrator.dataset_path

        # Create valid dataset
        atoms = [Atoms("H")]
        orchestrator.dataset_manager.save_iter(iter(atoms), dataset_path)

        # Verify it loads
        assert list(orchestrator.dataset_manager.load_iter(dataset_path))

        # Corrupt checksum file
        checksum_path = dataset_path.with_suffix(dataset_path.suffix + ".sha256")
        assert checksum_path.exists()
        checksum_path.write_text("invalid_hash")

        # Verify loading raises ValueError
        with pytest.raises(ValueError, match="Checksum verification failed"):
            list(orchestrator.dataset_manager.load_iter(dataset_path))

    def test_partial_write_recovery(self, error_config: PYACEMAKERConfig, tmp_path: Path) -> None:
        """Test behavior when dataset file is truncated/corrupted."""
        orchestrator = Orchestrator(error_config, base_dir=tmp_path)
        dataset_path = orchestrator.dataset_path

        # Write valid frame
        atoms = Atoms("H")
        obj_bytes = pickle.dumps(atoms)
        size = len(obj_bytes)

        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(dataset_path, "wb") as f:
            f.write(struct.pack(">Q", size))
            f.write(obj_bytes)
            # Write partial header (truncated)
            f.write(b"tru")

        # Load iter should return valid items and stop on corruption (logging error)
        loaded = list(orchestrator.dataset_manager.load_iter(dataset_path, verify=False))
        assert len(loaded) == 1
        assert loaded[0].get_chemical_formula() == "H"  # type: ignore[no-untyped-call]

    def test_orchestrator_dataset_corruption_handling(self, error_config: PYACEMAKERConfig, tmp_path: Path) -> None:
        """Test Orchestrator resilience to dataset corruption during cycle."""
        orchestrator = Orchestrator(error_config, base_dir=tmp_path)

        # Corrupt dataset (invalid gzip)
        dataset_path = orchestrator.dataset_path
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_path.write_bytes(b"garbage_data_not_gzip")

        # Create dummy potential so we don't fail on "No potential"
        orchestrator.current_potential = MagicMock()

        # Mock trainer to avoid other errors
        with patch("pyacemaker.modules.trainer.PacemakerTrainer.train"):
             result = orchestrator.run()

        # Should fail gracefully because gzip.open raises error,
        # _run_training_phase catches it, returns False -> Cycle Failed.
        from pyacemaker.domain_models.models import CycleStatus
        assert result.status == CycleStatus.FAILED
