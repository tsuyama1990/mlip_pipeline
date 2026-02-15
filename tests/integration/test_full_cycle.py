"""Integration tests for the full active learning cycle."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from ase import Atoms

from pyacemaker.core.config import (
    DFTConfig,
    DynamicsEngineConfig,
    OracleConfig,
    OrchestratorConfig,
    ProjectConfig,
    PYACEMAKERConfig,
    StructureGeneratorConfig,
    TrainerConfig,
)
from pyacemaker.domain_models.models import (
    HaltInfo,
    Potential,
    StructureMetadata,
    UncertaintyState,
)
from pyacemaker.modules.oracle import MockOracle
from pyacemaker.orchestrator import Orchestrator


class TestFullCycleIntegration:
    """End-to-end integration tests for the full cycle."""

    def test_full_cycle_execution(self, tmp_path: Path) -> None:
        """Test full execution of the orchestrator loop with mocks."""
        # Create dummy pseudopotential
        pp_path = tmp_path / "Fe.upf"
        pp_path.touch()

        config = PYACEMAKERConfig(
            version="0.1.0",
            project=ProjectConfig(name="IntegrationTest", root_dir=tmp_path),
            orchestrator=OrchestratorConfig(
                max_cycles=2,
                n_local_candidates=5,
                n_active_set_select=2,
            ),
            structure_generator=StructureGeneratorConfig(strategy="random"),
            oracle=OracleConfig(
                dft=DFTConfig(code="quantum_espresso", pseudopotentials={"Fe": str(pp_path)}),
                mock=True,
            ),
            trainer=TrainerConfig(mock=True),
            dynamics_engine=DynamicsEngineConfig(mock=True),
        )

        with (
            patch("pyacemaker.modules.trainer.PacemakerTrainer.train") as mock_train,
            patch("pyacemaker.modules.trainer.PacemakerTrainer.select_active_set") as mock_select,
            patch("pyacemaker.modules.dynamics_engine.MDInterface.run_md") as mock_run_md,
            # We also need to ensure the random check passes to trigger run_md call effectively
            # or just force run_exploration to yield something if we mocked run_exploration.
            # But we are testing integration, so let's mock run_md.
            # However, LAMMPSEngine.run_exploration has a random check.
            patch("pyacemaker.modules.dynamics_engine.secrets.SystemRandom") as mock_random,
        ):
            # Force random check to trigger halt logic in run_exploration loop
            # The loop: if secrets.SystemRandom().random() < float(probability):
            # We want this to be True (random < 0.3)
            mock_random.return_value.random.return_value = 0.1
            # MockOracle uses uniform, so we need to return a float to avoid MagicMock in energy
            mock_random.return_value.uniform.return_value = 0.5

            # Setup MDInterface Mock
            s = StructureMetadata(features={"atoms": Atoms("Fe")})
            s.uncertainty_state = UncertaintyState(gamma_max=10.0)
            mock_run_md.return_value = HaltInfo(halted=True, step=10, max_gamma=10.0, structure=s)

            # Setup Trainer Mocks
            mock_potential = MagicMock()
            mock_potential.path = tmp_path / "mock.yace"
            mock_train.return_value = mock_potential

            mock_active_set = MagicMock()
            mock_active_set.structures = [
                StructureMetadata(features={"atoms": Atoms("Fe")}),
                StructureMetadata(features={"atoms": Atoms("Fe")}),
            ]

            mock_select.return_value = mock_active_set

            orchestrator = Orchestrator(config)

            # Inject MockOracle directly to ensure it behaves as expected
            orchestrator.oracle = MockOracle(config)

            # Run
            result = orchestrator.run()

            assert result.status == "success"
            assert orchestrator.cycle_count == 2
            assert orchestrator.dataset_path.exists()

            # Verify interactions
            assert mock_train.call_count >= 2
            assert mock_select.call_count >= 2

    def test_cycle_failure_validation(self, tmp_path: Path) -> None:
        """Test that validation failure halts the cycle."""
        pp_path = tmp_path / "Fe.upf"
        pp_path.touch()

        config = PYACEMAKERConfig(
            version="0.1.0",
            project=ProjectConfig(name="FailureTest", root_dir=tmp_path),
            orchestrator=OrchestratorConfig(max_cycles=1, validation_split=1.0),
            structure_generator=StructureGeneratorConfig(strategy="random"),
            oracle=OracleConfig(
                dft=DFTConfig(code="qe", pseudopotentials={"Fe": str(pp_path)}), mock=True
            ),
            trainer=TrainerConfig(mock=True),
        )

        with (
            patch("pyacemaker.modules.validator.Validator.validate") as mock_validate,
            patch("pyacemaker.modules.trainer.PacemakerTrainer.train") as mock_train,
        ):
            # Simulate validation failure
            mock_validate.return_value = MagicMock(status="failed", metrics={})

            # Mock train must consume stream to populate validation set
            def consume_stream(dataset: Any, potential: Potential | None = None) -> Any:
                for _ in dataset:
                    pass
                return MagicMock()

            mock_train.side_effect = consume_stream

            orchestrator = Orchestrator(config)
            # Pre-populate dataset so training runs
            dataset_path = orchestrator.dataset_path
            orchestrator.dataset_manager.save_iter(iter([Atoms("Fe")]), dataset_path)

            result = orchestrator.run()

            # Should fail
        assert result.status == "FAILED"
        # Cycle count should be 1 (failed at cycle 1)
        assert orchestrator.cycle_count == 1
