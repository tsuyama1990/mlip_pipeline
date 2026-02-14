"""Tests for Adaptive Structure Generator."""

from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from pyacemaker.core.config import PYACEMAKERConfig, StructureGeneratorConfig
from pyacemaker.domain_models.models import StructureMetadata, UncertaintyState
from pyacemaker.generator.policy import ExplorationContext
from pyacemaker.modules.structure_generator import AdaptiveStructureGenerator


class TestAdaptiveStructureGenerator:
    @pytest.fixture
    def config(self) -> MagicMock:
        mock_config = MagicMock(spec=PYACEMAKERConfig)
        # We need actual model instance because AdaptiveStructureGenerator reads config
        # But we mock policy so it shouldn't matter much.
        # But to be safe, use basic structure.
        mock_config.structure_generator = StructureGeneratorConfig(strategy="adaptive")
        return mock_config

    @pytest.fixture
    def generator(self, config: MagicMock) -> AdaptiveStructureGenerator:
        # Patch AdaptivePolicy so we can spy on decide_strategy
        with patch("pyacemaker.modules.structure_generator.AdaptivePolicy") as MockPolicy:
            gen = AdaptiveStructureGenerator(config)
            # Attach mock policy to instance for verification
            gen.mock_policy = MockPolicy.return_value  # type: ignore[attr-defined]
            return gen

    def test_generate_local_candidates_cycle(
        self, generator: AdaptiveStructureGenerator
    ) -> None:
        """Test that cycle number is passed correctly to policy."""
        atoms = Atoms("H")
        seed = StructureMetadata(
            features={"atoms": atoms}, uncertainty_state=UncertaintyState(gamma_max=5.0)
        )

        # Mock strategy
        mock_strategy = MagicMock()
        mock_strategy.generate.return_value = [atoms.copy()]  # type: ignore[no-untyped-call]
        generator.mock_policy.decide_strategy.return_value = mock_strategy  # type: ignore[attr-defined]

        # Call with cycle=5
        list(generator.generate_local_candidates(seed, n_candidates=1, cycle=5))

        # Verify context
        call_args = generator.mock_policy.decide_strategy.call_args  # type: ignore[attr-defined]
        context = call_args[0][0]
        assert isinstance(context, ExplorationContext)
        assert context.cycle == 5
        assert context.seed_structure is seed

    def test_generate_batch_candidates_cycle(
        self, generator: AdaptiveStructureGenerator
    ) -> None:
        """Test that cycle is propagated in batch mode."""
        atoms = Atoms("H")
        seeds = [StructureMetadata(features={"atoms": atoms}, uncertainty_state=UncertaintyState())]

        mock_strategy = MagicMock()
        mock_strategy.generate.return_value = [atoms.copy()]  # type: ignore[no-untyped-call]
        generator.mock_policy.decide_strategy.return_value = mock_strategy  # type: ignore[attr-defined]

        list(generator.generate_batch_candidates(seeds, n_candidates_per_seed=1, cycle=10))

        # Verify context
        call_args = generator.mock_policy.decide_strategy.call_args  # type: ignore[attr-defined]
        context = call_args[0][0]
        assert context.cycle == 10
