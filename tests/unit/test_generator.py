import pytest
from ase import Atoms
from pathlib import Path
import numpy as np

from mlip_autopipec.domain_models.config import GeneratorConfig, ExplorationPolicyConfig
from mlip_autopipec.domain_models.enums import GeneratorType
from mlip_autopipec.domain_models.datastructures import Structure

# Imports that will be available later
try:
    from mlip_autopipec.generator.random_gen import RandomGenerator
except ImportError:
    RandomGenerator = None

try:
    from mlip_autopipec.generator.m3gnet_gen import M3GNetGenerator
except ImportError:
    M3GNetGenerator = None

try:
    from mlip_autopipec.generator.adaptive import AdaptiveGenerator
except ImportError:
    AdaptiveGenerator = None

class TestRandomGenerator:
    def test_random_generator_import(self):
        assert RandomGenerator is not None, "RandomGenerator not implemented"

    def test_random_generation(self, tmp_path):
        if RandomGenerator is None:
            pytest.fail("RandomGenerator not implemented")

        # Create a dummy seed file
        seed_path = tmp_path / "seed.xyz"
        atoms = Atoms("MgO", positions=[[0, 0, 0], [2, 0, 0]], cell=[4, 4, 4], pbc=True)
        atoms.write(seed_path)

        config = GeneratorConfig(
            type=GeneratorType.RANDOM,
            seed_structure_path=seed_path,
            policy=ExplorationPolicyConfig(strain_range=0.1)
        )

        generator = RandomGenerator(config)
        candidates = list(generator.explore({"count": 5}))

        assert len(candidates) == 5
        for s in candidates:
            assert s.provenance == "random"
            assert s.atoms.get_chemical_symbols() == ["Mg", "O"]

        # Check diversity
        pos0 = candidates[0].atoms.positions
        pos1 = candidates[1].atoms.positions
        assert not np.allclose(pos0, pos1), "Structures should be different"

    def test_random_generator_no_seed(self):
        if RandomGenerator is None:
            pytest.fail("RandomGenerator not implemented")

        config = GeneratorConfig(type=GeneratorType.RANDOM, seed_structure_path=None)
        with pytest.raises(ValueError):
             RandomGenerator(config)

class TestM3GNetGenerator:
    def test_m3gnet_generator_import(self):
        assert M3GNetGenerator is not None, "M3GNetGenerator not implemented"

    def test_m3gnet_generation(self):
        if M3GNetGenerator is None:
            pytest.fail("M3GNetGenerator not implemented")

        config = GeneratorConfig(type=GeneratorType.M3GNET)
        generator = M3GNetGenerator(config)

        candidates = list(generator.explore({"count": 2}))
        assert len(candidates) > 0
        assert candidates[0].provenance == "m3gnet"

class TestAdaptiveGenerator:
    def test_adaptive_generator_import(self):
        assert AdaptiveGenerator is not None, "AdaptiveGenerator not implemented"

    def test_temperature_schedule_explore(self):
        if AdaptiveGenerator is None:
            pytest.fail("AdaptiveGenerator not implemented")

        # Test explore logic picking up schedule
        policy = ExplorationPolicyConfig(temperature_schedule=[100.0, 200.0])
        config = GeneratorConfig(type=GeneratorType.ADAPTIVE, policy=policy)
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

    def test_adaptive_generator_with_seed(self, tmp_path):
        if AdaptiveGenerator is None:
            pytest.fail("AdaptiveGenerator not implemented")

        seed_path = tmp_path / "seed.xyz"
        atoms = Atoms("He", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
        atoms.write(seed_path)

        config = GeneratorConfig(
            type=GeneratorType.ADAPTIVE,
            seed_structure_path=seed_path,
            policy=ExplorationPolicyConfig(strain_range=0.1)
        )
        generator = AdaptiveGenerator(config)

        # Should use RandomGenerator internally
        assert generator.random_gen is not None

        candidates = list(generator.explore({"cycle": 0, "count": 1}))
        assert len(candidates) == 1
        assert candidates[0].provenance == "md_300.0K" # Default schedule 300K

    def test_generate_lammps_input(self):
        if AdaptiveGenerator is None:
            pytest.fail("AdaptiveGenerator not implemented")

        config = GeneratorConfig(type=GeneratorType.ADAPTIVE)
        generator = AdaptiveGenerator(config)

        script = generator._generate_lammps_input(temperature=500.0, steps=2000)
        assert "run 2000" in script
        assert "temp 500.0 500.0" in script
        assert "units metal" in script
