"""Tests for Structure Generator module."""

from pathlib import Path

import pytest

from pyacemaker.core.config import (
    DFTConfig,
    OracleConfig,
    ProjectConfig,
    PYACEMAKERConfig,
    StructureGeneratorConfig,
)
from pyacemaker.domain_models.models import StructureMetadata
from pyacemaker.modules.structure_generator import RandomStructureGenerator


@pytest.fixture
def config(tmp_path: Path) -> PYACEMAKERConfig:
    """Return a valid configuration."""
    return PYACEMAKERConfig(
        version="0.1.0",
        project=ProjectConfig(name="Test", root_dir=tmp_path),
        oracle=OracleConfig(
            dft=DFTConfig(
                code="quantum_espresso",
                pseudopotentials={"Fe": "Fe.pbe.UPF"},
            ),
            mock=True,
        ),
        structure_generator=StructureGeneratorConfig(strategy="random"),
    )


def test_random_generator_initial(config: PYACEMAKERConfig) -> None:
    """Test generating initial structures."""
    generator = RandomStructureGenerator(config)

    # Now returns list, but internally uses generator
    structures = generator.generate_initial_structures()

    assert isinstance(structures, list)
    assert len(structures) == 5
    assert isinstance(structures[0], StructureMetadata)
    assert structures[0].material_dna is not None


def test_random_generator_local(config: PYACEMAKERConfig) -> None:
    """Test generating local candidates."""
    generator = RandomStructureGenerator(config)
    seed = StructureMetadata(tags=["seed"])

    candidates = generator.generate_local_candidates(seed, n_candidates=3)

    assert len(candidates) == 3
    assert candidates[0].tags == ["candidate", "local"]


def test_random_generator_batch(config: PYACEMAKERConfig) -> None:
    """Test generating batch candidates."""
    generator = RandomStructureGenerator(config)
    seeds = [StructureMetadata(tags=["seed1"]), StructureMetadata(tags=["seed2"])]

    candidates = generator.generate_batch_candidates(seeds, n_candidates_per_seed=2)

    assert len(candidates) == 4 # 2 * 2
    assert candidates[0].tags == ["candidate", "batch"]
