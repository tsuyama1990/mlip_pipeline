"""Tests for Validator module."""

from pathlib import Path
from unittest.mock import patch

import pytest
from ase import Atoms

from pyacemaker.core.config import (
    DFTConfig,
    OracleConfig,
    ProjectConfig,
    PYACEMAKERConfig,
    ValidatorConfig,
)
from pyacemaker.domain_models.models import (
    Potential,
    PotentialType,
    StructureMetadata,
)
from pyacemaker.domain_models.validator import ValidationResult
from pyacemaker.modules.validator import Validator


@pytest.fixture
def config(tmp_path: Path) -> PYACEMAKERConfig:
    """Return a valid configuration."""
    return PYACEMAKERConfig(
        version="0.1.0",
        project=ProjectConfig(name="test", root_dir=tmp_path),
        validator=ValidatorConfig(),
        oracle=OracleConfig(
            dft=DFTConfig(pseudopotentials={"H": "H.upf"}), mock=True
        ),
    )


def test_validator_single_pass_selection(config: PYACEMAKERConfig) -> None:
    """Test that Validator selects the best structure in a single pass."""
    validator = Validator(config)

    # Create test set
    # Structure 1: Energy -5.0, 2 atoms -> -2.5 eV/atom
    # Structure 2: Energy -6.0, 2 atoms -> -3.0 eV/atom (Best)
    # Structure 3: Energy -2.0, 1 atom -> -2.0 eV/atom

    s1 = StructureMetadata(energy=-5.0, features={"atoms": Atoms("H2")})
    s2 = StructureMetadata(energy=-6.0, features={"atoms": Atoms("H2")})
    s3 = StructureMetadata(energy=-2.0, features={"atoms": Atoms("H")})

    def test_stream():
        yield s1
        yield s2
        yield s3

    potential = Potential(
        path=Path("dummy.yace"),
        type=PotentialType.PACE,
        version="1.0",
        metrics={},
        parameters={},
    )

    with patch("pyacemaker.modules.validator.ValidatorManager") as MockManager:
        mock_manager_instance = MockManager.return_value
        mock_manager_instance.validate.return_value = ValidationResult(
            passed=True,
            metrics={"bulk_modulus": 100.0},
            artifacts={},
            phonon_stable=True,
            elastic_stable=True,
        )

        result = validator.validate(potential, test_stream())

        assert result.status == "success"
        # Verify manager was called with s2 (best structure)
        # Note: We need to check call args.
        # But wait, Validator passes `reference_structure` which is an Atoms object.
        # s2.features["atoms"] is the object.

        # Check call args
        mock_manager_instance.validate.assert_called_once()
        call_kwargs = mock_manager_instance.validate.call_args.kwargs

        # Check potential passed
        assert call_kwargs["potential"] == potential

        # Check structure passed (s2's atoms)
        # Atoms equality check might be tricky if reference identity is lost (it shouldn't be)
        passed_structure = call_kwargs["structure"]
        assert passed_structure is s2.features["atoms"]

        # Check metrics
        assert result.metrics.rmse_energy == 0.0
        assert result.metrics.count == 3
