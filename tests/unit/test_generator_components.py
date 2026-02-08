from typing import Any

import numpy as np
from ase import Atoms

from mlip_autopipec.components.generator.adaptive import AdaptiveGenerator
from mlip_autopipec.components.generator.builder import BulkBuilder, SurfaceBuilder
from mlip_autopipec.components.generator.policy import ExplorationPolicy
from mlip_autopipec.components.generator.rattle import RattleTransform, StrainTransform
from mlip_autopipec.domain_models.config import AdaptiveGeneratorConfig
from mlip_autopipec.domain_models.enums import GeneratorType
from mlip_autopipec.domain_models.structure import Structure


class TestBulkBuilder:
    def test_build_bulk(self) -> None:
        config = AdaptiveGeneratorConfig(
            name=GeneratorType.ADAPTIVE,
            element="Fe",
            crystal_structure="bcc",
            strain_range=0.0,
            rattle_strength=0.0,
        )
        builder = BulkBuilder()
        structures = list(builder.build(n_structures=5, config=config))

        assert len(structures) == 5
        for s in structures:
            assert isinstance(s, Structure)
            assert s.tags.get("type") == "bulk"
            assert len(s.atomic_numbers) > 0
            # Check if it's Fe (atomic number 26)
            assert np.all(s.atomic_numbers == 26)


class TestSurfaceBuilder:
    def test_build_surface(self) -> None:
        config = AdaptiveGeneratorConfig(
            name=GeneratorType.ADAPTIVE,
            element="Fe",
            crystal_structure="bcc",
            surface_indices=[[1, 0, 0]],
            vacuum=10.0,
        )
        builder = SurfaceBuilder()
        structures = list(builder.build(n_structures=2, config=config))

        assert len(structures) == 2
        for s in structures:
            assert isinstance(s, Structure)
            assert s.tags.get("type") == "surface"
            # Check vacuum (z-axis should be large)
            assert s.cell[2, 2] > 10.0


class TestRattleTransform:
    def test_apply_rattle(self) -> None:
        # Create a simple structure
        atoms = Atoms("H", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
        structure = Structure.from_ase(atoms)
        original_pos = structure.positions.copy()

        transform = RattleTransform(stdev=0.1)
        transformed = transform.apply(structure)

        assert not np.allclose(transformed.positions, original_pos)
        assert np.allclose(transformed.cell, structure.cell)


class TestStrainTransform:
    def test_apply_strain(self) -> None:
        atoms = Atoms("H", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
        structure = Structure.from_ase(atoms)
        original_cell = structure.cell.copy()

        transform = StrainTransform(strain_range=0.1)
        transformed = transform.apply(structure)

        assert not np.allclose(transformed.cell, original_cell)
        # Volume should change (unless pure shear, but random strain usually changes volume)
        # Check that it is still a valid cell


class TestExplorationPolicy:
    def test_decide_next_batch_cold_start(self) -> None:
        policy = ExplorationPolicy()
        metrics: dict[str, Any] = {}  # Empty metrics -> Cold start
        tasks = policy.decide_next_batch(current_cycle=0, current_metrics=metrics, n_total=10)

        # Expect bulk and maybe some surfaces
        assert len(tasks) > 0
        total_structures = sum(t.n_structures for t in tasks)
        assert total_structures == 10

        # Check task types
        task_types = [t.builder_name for t in tasks]
        assert "bulk" in task_types

    def test_decide_next_batch_high_surface_error(self) -> None:
        policy = ExplorationPolicy()
        metrics = {"validation_error": {"surface": 0.5, "bulk": 0.01}}
        tasks = policy.decide_next_batch(current_cycle=1, current_metrics=metrics, n_total=10)

        # Should prioritize surface
        surface_tasks = [t for t in tasks if t.builder_name == "surface"]
        assert len(surface_tasks) > 0
        n_surface = sum(t.n_structures for t in surface_tasks)
        assert n_surface >= 5  # At least 50%


class TestAdaptiveGenerator:
    def test_generate(self) -> None:
        config = AdaptiveGeneratorConfig(
            name=GeneratorType.ADAPTIVE,
            element="Fe",
            crystal_structure="bcc",
            # policy_ratios uses default factory which is reasonable
            n_structures=10
        )
        generator = AdaptiveGenerator(config)

        # Call generate with n_structures=10
        structures = list(generator.generate(n_structures=10))

        assert len(structures) == 10

        types = [s.tags.get("type") for s in structures]
        assert "bulk" in types
        # Surface ratio is 0.4 by default, so we expect some surfaces
        assert "surface" in types
