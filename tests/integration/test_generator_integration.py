
from mlip_autopipec.components.generator.adaptive import AdaptiveGenerator
from mlip_autopipec.domain_models.config import GeneratorConfig
from mlip_autopipec.domain_models.structure import Structure


class TestGeneratorIntegration:
    def test_adaptive_generator_flow(self) -> None:
        # Create config
        config = GeneratorConfig(
            name="adaptive",
            element="Fe",
            crystal_structure="bcc",
            strain_range=0.05,
            rattle_strength=0.01,
            surface_indices=[[1, 0, 0]],
            vacuum=10.0,
            n_structures=10,
        )

        # Instantiate Generator
        generator = AdaptiveGenerator(config)

        # Generate structures
        structures = list(generator.generate(n_structures=10))

        assert len(structures) == 10
        assert all(isinstance(s, Structure) for s in structures)

        # Check that we have valid structures
        # Should have some bulk and some surface (by default policy)
        bulk_count = sum(1 for s in structures if s.tags.get("type") == "bulk")
        surface_count = sum(1 for s in structures if s.tags.get("type") == "surface")

        assert bulk_count > 0
        assert surface_count > 0

        # Check properties
        s0 = structures[0]
        assert s0.positions.shape[1] == 3
        assert s0.atomic_numbers[0] == 26  # Fe
