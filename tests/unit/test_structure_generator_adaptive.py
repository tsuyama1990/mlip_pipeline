"""Tests for AdaptiveStructureGenerator."""

from pathlib import Path

import pytest

from pyacemaker.core.config import (
    CONSTANTS,
    DFTConfig,
    OracleConfig,
    ProjectConfig,
    PYACEMAKERConfig,
    StructureGeneratorConfig,
)
from pyacemaker.domain_models.models import (
    PredictedProperties,
    StructureMetadata,
    UncertaintyState,
)
from pyacemaker.modules.structure_generator import AdaptiveStructureGenerator


@pytest.fixture
def mock_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> PYACEMAKERConfig:
    """Create a minimal mock configuration."""
    monkeypatch.setattr(CONSTANTS, "skip_file_checks", True)
    return PYACEMAKERConfig(
        version="0.1.0",
        project=ProjectConfig(name="test", root_dir=tmp_path),
        oracle=OracleConfig(dft=DFTConfig(pseudopotentials={"H": "H.upf"})),
        structure_generator=StructureGeneratorConfig(strategy="adaptive"),
    )


def test_determine_policy_default(mock_config: PYACEMAKERConfig) -> None:
    """Test default policy when no special features present."""
    gen = AdaptiveStructureGenerator(mock_config)
    s = StructureMetadata()
    policy = gen._determine_policy(s)
    assert policy["mode"] == "default"


def test_determine_policy_defect(mock_config: PYACEMAKERConfig) -> None:
    """Test defect-driven policy for insulators (band gap > 0)."""
    gen = AdaptiveStructureGenerator(mock_config)
    s = StructureMetadata(
        predicted_properties=PredictedProperties(band_gap=1.0)
    )
    policy = gen._determine_policy(s)
    assert policy["mode"] == "defect_driven"
    assert policy["n_defects"] == 2


def test_determine_policy_cautious(mock_config: PYACEMAKERConfig) -> None:
    """Test cautious policy for high uncertainty."""
    gen = AdaptiveStructureGenerator(mock_config)
    s = StructureMetadata(
        uncertainty_state=UncertaintyState(gamma_max=10.0)
    )
    policy = gen._determine_policy(s)
    assert policy["mode"] == "cautious"
    assert policy["temperature"] == 100.0
