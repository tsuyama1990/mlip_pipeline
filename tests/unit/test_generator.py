from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from mlip_autopipec.domain_models.config import ExplorationPolicyConfig, GeneratorConfig
from mlip_autopipec.domain_models.enums import GeneratorType
from mlip_autopipec.generator.adaptive import AdaptiveGenerator
from mlip_autopipec.generator.m3gnet_gen import M3GNetGenerator

# Imports that will be available later
from mlip_autopipec.generator.random_gen import RandomGenerator


class TestRandomGenerator:
    def test_random_generator_import(self) -> None:
        assert RandomGenerator is not None, "RandomGenerator not implemented"

    def test_random_generation(self, tmp_path: Path) -> None:
        # Create a dummy seed file
        seed_path = tmp_path / "seed.xyz"
        atoms = Atoms("MgO", positions=[[0, 0, 0], [2, 0, 0]], cell=[4, 4, 4], pbc=True)
        atoms.write(seed_path)  # type: ignore[no-untyped-call]

        config = GeneratorConfig(
            type=GeneratorType.RANDOM,
            seed_structure_path=seed_path,
            policy=ExplorationPolicyConfig(strain_range=0.1),
        )

        generator = RandomGenerator(config)
        candidates = list(generator.explore({"count": 5}))

        assert len(candidates) == 5
        for s in candidates:
            assert s.provenance == "random"
            atoms_obj = s.ase_atoms
            assert atoms_obj.get_chemical_symbols() == ["Mg", "O"]  # type: ignore[no-untyped-call]

        # Check diversity
        pos0 = candidates[0].ase_atoms.positions
        pos1 = candidates[1].ase_atoms.positions
        assert not np.allclose(pos0, pos1), "Structures should be different"

    def test_random_generator_no_seed(self) -> None:
        config = GeneratorConfig(type=GeneratorType.RANDOM, seed_structure_path=None)
        with pytest.raises(
            ValueError, match="RandomGenerator requires a seed structure path in config"
        ):
            RandomGenerator(config)

    def test_random_generator_invalid_seed(self, tmp_path: Path) -> None:
        """Test behavior when seed file exists but is empty/invalid."""
        seed_path = tmp_path / "empty.xyz"
        seed_path.touch()

        config = GeneratorConfig(type=GeneratorType.RANDOM, seed_structure_path=seed_path)
        generator = RandomGenerator(config)

        # We expect a ValueError now, specifically about reading failure
        with pytest.raises(ValueError, match="Failed to read seed structure"):
            list(generator.explore({"count": 1}))

    def test_malformed_seed(self, tmp_path: Path) -> None:
        """Test behavior when seed file has malformed content."""
        seed_path = tmp_path / "malformed.xyz"
        seed_path.write_text("This is not a valid XYZ file")

        config = GeneratorConfig(type=GeneratorType.RANDOM, seed_structure_path=seed_path)
        generator = RandomGenerator(config)

        # Should raise ValueError wrapping the underlying ASE error
        with pytest.raises(ValueError, match="Failed to read seed structure"):
            list(generator.explore({"count": 1}))


class TestM3GNetGenerator:
    def test_m3gnet_generator_import(self) -> None:
        assert M3GNetGenerator is not None, "M3GNetGenerator not implemented"

    def test_m3gnet_generation(self) -> None:
        config = GeneratorConfig(type=GeneratorType.M3GNET)
        generator = M3GNetGenerator(config)

        candidates = list(generator.explore({"count": 2}))
        assert len(candidates) > 0
        assert candidates[0].provenance == "m3gnet"


class TestAdaptiveGenerator:
    def test_adaptive_generator_import(self) -> None:
        assert AdaptiveGenerator is not None, "AdaptiveGenerator not implemented"

    def test_temperature_schedule_explore(self, tmp_path: Path) -> None:
        # Need a seed for mock execution now
        seed_path = tmp_path / "seed_adapt.xyz"
        Atoms("He", positions=[[0, 0, 0]], cell=[5, 5, 5]).write(seed_path)  # type: ignore[no-untyped-call]

        # Test explore logic picking up schedule
        policy = ExplorationPolicyConfig(temperature_schedule=[100.0, 200.0])
        config = GeneratorConfig(
            type=GeneratorType.ADAPTIVE, policy=policy, seed_structure_path=seed_path
        )
        generator = AdaptiveGenerator(config)

        # Cycle 0 -> 100K
        candidates_c0 = list(generator.explore({"cycle": 0, "count": 1}))
        assert len(candidates_c0) == 1
        assert candidates_c0[0].provenance == "md_100.0K"

        # Cycle 1 -> 200K
        candidates_c1 = list(generator.explore({"cycle": 1, "count": 1}))
        assert len(candidates_c1) == 1
        assert candidates_c1[0].provenance == "md_200.0K"

        # Cycle 2 -> Clamped to 200K
        candidates_c2 = list(generator.explore({"cycle": 2, "count": 1}))
        assert len(candidates_c2) == 1
        assert candidates_c2[0].provenance == "md_200.0K"

    def test_adaptive_generator_with_seed(self, tmp_path: Path) -> None:
        seed_path = tmp_path / "seed.xyz"
        atoms = Atoms("He", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
        atoms.write(seed_path)  # type: ignore[no-untyped-call]

        config = GeneratorConfig(
            type=GeneratorType.ADAPTIVE,
            seed_structure_path=seed_path,
            policy=ExplorationPolicyConfig(strain_range=0.1),
        )
        generator = AdaptiveGenerator(config)

        # Should use RandomGenerator internally
        assert generator.random_gen is not None

        candidates = list(generator.explore({"cycle": 0, "count": 1}))
        assert len(candidates) == 1
        assert candidates[0].provenance == "md_300.0K"  # Default schedule 300K

    def test_generate_lammps_input(self) -> None:
        config = GeneratorConfig(type=GeneratorType.ADAPTIVE)
        generator = AdaptiveGenerator(config)

        with generator._lammps_input_context(temperature=500.0, steps=2000) as path:
            assert path.exists()
            content = path.read_text()
            assert "run 2000" in content
            assert "temp 500.0 500.0" in content
            assert "units metal" in content
            # Cleanup happens on exit

        assert not path.exists()

    def test_adaptive_generator_no_seed_fallback(self) -> None:
        """Test that AdaptiveGenerator falls back to M3GNet when no seed is provided."""
        config = GeneratorConfig(type=GeneratorType.ADAPTIVE, seed_structure_path=None)
        generator = AdaptiveGenerator(config)

        # Should not have random_gen
        assert generator.random_gen is None
        # Should have m3gnet_gen
        assert generator.m3gnet_gen is not None

        # Explore should return M3GNet structures
        candidates = list(generator.explore({"cycle": 0, "count": 1}))
        assert len(candidates) == 1
        assert candidates[0].provenance == "m3gnet"
