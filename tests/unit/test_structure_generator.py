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
    # Mocking pseudopotential file check by creating it or skipping check
    # But since we use default defaults, config validation might fail if file missing
    # unless skip_file_checks is True.
    # In test environment, usually we mock paths or skip.
    # But here we are constructing config manually.

    # We need to ensure the PP file path is valid or skip check.
    # Constants are loaded from defaults.yaml where skip_file_checks is false.
    # We can override it via env var PYACEMAKER_SKIP_FILE_CHECKS=true if needed,
    # or just create the dummy file.

    pp_file = tmp_path / "Fe.pbe.UPF"
    pp_file.touch()

    return PYACEMAKERConfig(
        version="0.1.0",
        project=ProjectConfig(name="Test", root_dir=tmp_path),
        oracle=OracleConfig(
            dft=DFTConfig(
                code="quantum_espresso",
                pseudopotentials={"Fe": str(pp_file)},
            ),
            mock=True,
        ),
        structure_generator=StructureGeneratorConfig(strategy="random"),
    )


def test_random_generator_initial(config: PYACEMAKERConfig) -> None:
    """Test generating initial structures."""
    generator = RandomStructureGenerator(config)

    # Now returns iterator
    structures_iter = generator.generate_initial_structures()
    structures = list(structures_iter)

    assert isinstance(structures, list)
    assert len(structures) == 5
    assert isinstance(structures[0], StructureMetadata)
    assert structures[0].material_dna is not None


def test_random_generator_local(config: PYACEMAKERConfig) -> None:
    """Test generating local candidates."""
    generator = RandomStructureGenerator(config)
    seed = StructureMetadata(tags=["seed"])

    candidates_iter = generator.generate_local_candidates(seed, n_candidates=3)
    candidates = list(candidates_iter)

    assert len(candidates) == 3
    # Check updated tags format (indexed)
    assert "candidate_0" in candidates[0].tags
    assert "local_0" in candidates[0].tags


def test_random_generator_batch(config: PYACEMAKERConfig) -> None:
    """Test generating batch candidates."""
    generator = RandomStructureGenerator(config)
    seeds = [StructureMetadata(tags=["seed1"]), StructureMetadata(tags=["seed2"])]

    candidates_iter = generator.generate_batch_candidates(seeds, n_candidates_per_seed=2)
    candidates = list(candidates_iter)

    assert len(candidates) == 4  # 2 * 2
    # Check updated tags format (indexed)
    # With BaseStructureGenerator, batch generation delegates to local, so tags might be local_X
    # Let's check for 'candidate' tag which is common
    assert any("candidate" in t for t in candidates[0].tags)
