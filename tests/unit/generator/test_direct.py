"""Tests for DirectGenerator."""

from pathlib import Path

import numpy as np
import pytest

from pyacemaker.core.config import (
    CONSTANTS,
    DFTConfig,
    OracleConfig,
    ProjectConfig,
    PYACEMAKERConfig,
    StructureGeneratorConfig,
)
from pyacemaker.domain_models.models import StructureMetadata, StructureStatus
from pyacemaker.generator.direct import DirectGenerator


@pytest.fixture
def mock_config(tmp_path: Path) -> PYACEMAKERConfig:
    """Mock configuration."""
    project_config = ProjectConfig(name="test", root_dir=tmp_path)
    oracle_config = OracleConfig(
        dft=DFTConfig(pseudopotentials={"Fe": "Fe.pbe"}),
        mock=True,
    )

    config = PYACEMAKERConfig(
        version=CONSTANTS.default_version,
        project=project_config,
        oracle=oracle_config,
    )
    # Setup Direct Generator Config
    config.structure_generator = StructureGeneratorConfig(
        strategy="direct",
        parameters={"dimensionality": 3, "bounds": [[0.0, 1.0]] * 3},
    )
    return config


def test_direct_generator_initialization(mock_config: PYACEMAKERConfig) -> None:
    """Test initialization of DirectGenerator."""
    generator = DirectGenerator(mock_config)
    assert generator.config.structure_generator.strategy == "direct"


def test_generate_direct_samples_count(mock_config: PYACEMAKERConfig) -> None:
    """Test that generate_direct_samples produces the correct number of samples."""
    generator = DirectGenerator(mock_config)
    samples = list(generator.generate_direct_samples(n_samples=10))
    assert len(samples) == 10
    assert all(isinstance(s, StructureMetadata) for s in samples)
    assert all(s.generation_method == "direct" for s in samples)


def test_generate_direct_samples_diversity(mock_config: PYACEMAKERConfig) -> None:
    """Test that generated samples are diverse (not identical)."""
    generator = DirectGenerator(mock_config)
    samples = list(generator.generate_direct_samples(n_samples=20))

    # Check pairwise distances of positions
    positions = []
    for s in samples:
        atoms = s.features.get("atoms")
        assert atoms is not None
        positions.append(atoms.get_positions().flatten())

    positions_array = np.array(positions)

    # Calculate pairwise distances
    from scipy.spatial.distance import pdist
    distances = pdist(positions_array)

    # Assert mean distance is significant (not all zero)
    assert np.mean(distances) > 0.1
    assert np.min(distances) > 0.0


def test_generate_direct_samples_objective(mock_config: PYACEMAKERConfig) -> None:
    """Test that objective affects generation (mock check)."""
    generator = DirectGenerator(mock_config)
    samples = list(generator.generate_direct_samples(n_samples=5, objective="maximize_entropy"))
    for s in samples:
        assert "objective:maximize_entropy" in s.tags


def test_generate_local_candidates(mock_config: PYACEMAKERConfig) -> None:
    """Test generation of local candidates (fallback)."""
    generator = DirectGenerator(mock_config)

    # Create a mock seed structure with atoms
    from ase import Atoms
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    seed = StructureMetadata(features={"atoms": atoms}, status=StructureStatus.NEW)

    candidates = list(generator.generate_local_candidates(seed, n_candidates=5))
    assert len(candidates) == 5
    assert all("local" in c.tags for c in candidates)

    # Check that positions are perturbed
    orig_pos = atoms.get_positions()  # type: ignore[no-untyped-call]
    new_pos = candidates[0].features["atoms"].get_positions()
    assert not np.allclose(orig_pos, new_pos)


def test_generate_batch_candidates(mock_config: PYACEMAKERConfig) -> None:
    """Test batch candidate generation."""
    generator = DirectGenerator(mock_config)

    from ase import Atoms
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    seed = StructureMetadata(features={"atoms": atoms}, status=StructureStatus.NEW)

    seeds = [seed] * 3
    candidates = list(generator.generate_batch_candidates(seeds, n_candidates_per_seed=2))
    assert len(candidates) == 6  # 3 seeds * 2 candidates


def test_run_and_initial(mock_config: PYACEMAKERConfig) -> None:
    """Test run and initial structures."""
    generator = DirectGenerator(mock_config)
    res = generator.run()
    assert res.status == "success"

    initial = list(generator.generate_initial_structures())
    assert len(initial) == 20


def test_strategy_info(mock_config: PYACEMAKERConfig) -> None:
    """Test strategy info."""
    generator = DirectGenerator(mock_config)
    info = generator.get_strategy_info()
    assert info["strategy"] == "direct_maxmin"
