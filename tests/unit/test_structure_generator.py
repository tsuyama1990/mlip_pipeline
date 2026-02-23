"""Unit tests for Structure Generator (Direct Sampling)."""

import pytest

from pyacemaker.core.config import StructureGeneratorConfig
from pyacemaker.domain_models.models import StructureMetadata

# Import from expected location (will fail until implemented)
try:
    from pyacemaker.generator.direct import DirectGenerator
except ImportError:
    DirectGenerator = None


@pytest.mark.skipif(DirectGenerator is None, reason="DirectGenerator not implemented")
def test_direct_generator_initialization() -> None:
    """Test initialization of DirectGenerator."""
    config = StructureGeneratorConfig(strategy="random")
    generator = DirectGenerator(config)
    assert generator.config == config


@pytest.mark.skipif(DirectGenerator is None, reason="DirectGenerator not implemented")
def test_generate_direct_samples_count() -> None:
    """Test that generate_direct_samples returns the requested number of samples."""
    config = StructureGeneratorConfig(strategy="random")
    generator = DirectGenerator(config)

    samples = list(generator.generate_direct_samples(n_samples=10))
    assert len(samples) == 10
    assert all(isinstance(s, StructureMetadata) for s in samples)


@pytest.mark.skipif(DirectGenerator is None, reason="DirectGenerator not implemented")
def test_generate_direct_samples_metadata() -> None:
    """Test metadata of generated samples."""
    config = StructureGeneratorConfig(strategy="random")
    generator = DirectGenerator(config)

    samples = list(generator.generate_direct_samples(n_samples=5))
    for s in samples:
        assert s.generation_method == "direct"
        assert s.status == "NEW"
        # Check if atoms are present in features
        assert "atoms" in s.features
        # Material DNA should be populated (composition)
        assert s.material_dna is not None
        assert s.material_dna.composition
