"""Tests for metadata preservation in Trainer module."""

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from ase import Atoms

from pyacemaker.core.config import PYACEMAKERConfig
from pyacemaker.core.utils import metadata_to_atoms
from pyacemaker.domain_models.models import MaterialDNA, StructureMetadata, UncertaintyState
from pyacemaker.modules.trainer import PacemakerTrainer


class TestTrainerMetadataPreservation:
    @pytest.fixture
    def config(self) -> MagicMock:
        mock_config = MagicMock(spec=PYACEMAKERConfig)
        # Mock the trainer config object properly
        # We need to mock the attributes accessed by __init__
        mock_config.trainer = MagicMock()
        mock_config.trainer.cutoff = 5.0
        mock_config.trainer.mock = False
        return mock_config

    @pytest.fixture
    def trainer(self, config: MagicMock) -> PacemakerTrainer:
        with (
            patch("pyacemaker.modules.trainer.PacemakerWrapper"),
            patch("pyacemaker.modules.trainer.ActiveSetSelector"),
            patch("pyacemaker.modules.trainer.DatasetManager"),
        ):
            return PacemakerTrainer(config)

    def test_metadata_roundtrip(self, trainer: PacemakerTrainer) -> None:
        """Test that metadata is preserved through atoms conversion."""
        # Create full metadata structure
        original_id = uuid4()
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        dna = MaterialDNA(composition={"H": 1.0})
        uncertainty = UncertaintyState(gamma_max=5.0)

        original = StructureMetadata(
            id=original_id,
            features={"atoms": atoms},
            material_dna=dna,
            uncertainty_state=uncertainty,
            energy=-13.6,
            forces=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            tags=["test_tag"],
        )

        # 1. Convert to Atoms
        atoms_converted = metadata_to_atoms(original)

        # Verify JSON injection
        assert "_metadata_json" in atoms_converted.info

        # 2. Simulate reconstruction
        json_str = atoms_converted.info["_metadata_json"]
        reconstructed = StructureMetadata.model_validate_json(json_str)
        reconstructed.features["atoms"] = atoms_converted  # re-attach atoms

        # Verify fidelity
        assert reconstructed.id == original.id
        assert reconstructed.material_dna == original.material_dna
        assert reconstructed.uncertainty_state == original.uncertainty_state
        assert reconstructed.tags == original.tags
        assert reconstructed.energy == original.energy
