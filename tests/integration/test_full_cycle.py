"""Integration tests for the full active learning cycle."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from ase import Atoms

from pyacemaker.core.config import (
    DFTConfig,
    OracleConfig,
    OrchestratorConfig,
    ProjectConfig,
    PYACEMAKERConfig,
    StructureGeneratorConfig,
    TrainerConfig,
)
from pyacemaker.domain_models.models import StructureMetadata
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
        )

        # Mock dependencies to avoid actual external calls
        # Mock Oracle is already a class we can use, but we need to ensure it works in this context
        # Mock Trainer needs to simulate returning a Potential and ActiveSet

        # We instantiate Orchestrator with real classes where possible (MockOracle, RandomStructureGenerator)
        # but patch Trainer since it calls subprocess.

        # We also need to patch DynamicsEngine because the default implementation
        # might not trigger 'halt' events deterministically with `secrets`.
        # We force `_simulate_halt_condition` to return True so we get candidate structures.
        with (
            patch("pyacemaker.modules.trainer.PacemakerTrainer.train") as mock_train,
            patch("pyacemaker.modules.trainer.PacemakerTrainer.select_active_set") as mock_select,
            patch(
                "pyacemaker.modules.dynamics_engine.LAMMPSEngine._simulate_halt_condition",
                return_value=True,
            ),
        ):
            # Setup Trainer Mocks
            mock_potential = MagicMock()
            mock_potential.path = tmp_path / "mock.yace"
            mock_train.return_value = mock_potential

            mock_active_set = MagicMock()
            mock_active_set.structures = [
                StructureMetadata(features={"atoms": Atoms("Fe")}),
                StructureMetadata(features={"atoms": Atoms("Fe")}),
            ]
            # Ensure IDs match candidates to simulate selection
            # But Orchestrator generates new candidates.
            # In mock mode, we just need select_active_set to return *something* that Orchestrator accepts.
            # Wait, Orchestrator filters by ID: `selected_structures = [c for c in candidates_list if c.id in active_ids]`
            # If we use streaming in Orchestrator, `select_active_set` in Trainer handles consumption.
            # Our updated Orchestrator logic:
            # `active_set = self.trainer.select_active_set(candidates_iter, ...)`
            # `if active_set.structures: selected_structures = active_set.structures`
            # So if we mock `select_active_set` to return an ActiveSet with `.structures`, we are good.

            mock_select.return_value = mock_active_set

            orchestrator = Orchestrator(config)

            # Inject MockOracle directly to ensure it behaves as expected
            orchestrator.oracle = MockOracle(config)

            # Run
            result = orchestrator.run()

            assert result.status == "success"
            assert orchestrator.cycle_count == 2
            assert orchestrator.dataset_path.exists()
            assert orchestrator.dataset_path.stat().st_size > 0

            # Verify interactions
            assert mock_train.call_count >= 2
            assert mock_select.call_count >= 2
