"""Orchestrator module implementation."""

from pyacemaker.core.base import Metrics, ModuleResult
from pyacemaker.core.config import PYACEMAKERConfig
from pyacemaker.core.interfaces import (
    DynamicsEngine,
    IOrchestrator,
    Oracle,
    StructureGenerator,
    Trainer,
    Validator,
)
from pyacemaker.domain_models.models import (
    CycleStatus,
    Potential,
    StructureMetadata,
)
from pyacemaker.modules.dynamics_engine import LAMMPSEngine
from pyacemaker.modules.oracle import MockOracle
from pyacemaker.modules.structure_generator import RandomStructureGenerator
from pyacemaker.modules.trainer import PacemakerTrainer
from pyacemaker.modules.validator import MockValidator


class Orchestrator(IOrchestrator):
    """Main Orchestrator for the active learning cycle."""

    def __init__(
        self,
        config: PYACEMAKERConfig,
        structure_generator: StructureGenerator | None = None,
        oracle: Oracle | None = None,
        trainer: Trainer | None = None,
        dynamics_engine: DynamicsEngine | None = None,
        validator: Validator | None = None,
    ) -> None:
        """Initialize the orchestrator and sub-modules.

        Dependencies can be injected; otherwise, default implementations are instantiated.
        """
        super().__init__(config)
        self.config = config

        # Dependency Injection with fallbacks
        self.structure_generator: StructureGenerator = (
            structure_generator or RandomStructureGenerator(config)
        )
        self.oracle: Oracle = oracle or MockOracle(config)
        self.trainer: Trainer = trainer or PacemakerTrainer(config)
        self.dynamics_engine: DynamicsEngine = dynamics_engine or LAMMPSEngine(config)
        self.validator: Validator = validator or MockValidator(config)

        # State
        self.current_potential: Potential | None = None
        self.dataset: list[StructureMetadata] = []
        self.cycle_count = 0

    def run(self) -> ModuleResult:
        """Run the full active learning pipeline."""
        self.logger.info("Starting Active Learning Pipeline")

        # 0. Cold Start (Initial Dataset)
        if not self.dataset:
            self.logger.info("Cold Start: Generating initial structures")
            initial_structures = self.structure_generator.generate_initial_structures()
            self.dataset.extend(self.oracle.compute_batch(initial_structures))

        # Main Loop
        max_cycles = self.config.orchestrator.max_cycles
        while self.cycle_count < max_cycles:
            self.cycle_count += 1
            self.logger.info(f"--- Cycle {self.cycle_count}/{max_cycles} ---")

            status = self.run_cycle()

            if status == CycleStatus.CONVERGED:
                self.logger.info("Convergence reached!")
                break
            if status == CycleStatus.FAILED:
                self.logger.error("Cycle failed!")
                return ModuleResult(
                    status="failed",
                    metrics=Metrics.model_validate({"cycles": self.cycle_count}),
                )

        self.logger.info("Pipeline completed")
        return ModuleResult(
            status="success",
            metrics=Metrics.model_validate(
                {"cycles": self.cycle_count, "dataset_size": len(self.dataset)}
            ),
        )

    def run_cycle(self) -> CycleStatus:
        """Execute one active learning cycle."""
        # 1. Training (Refinement)
        self.logger.info("Phase: Training")
        potential = self.trainer.train(self.dataset, self.current_potential)
        self.current_potential = potential

        # 2. Validation
        self.logger.info("Phase: Validation")
        # Split dataset for validation (simple holdout for now)
        # TODO: Implement better splitting strategy (e.g., config based)
        test_size = max(1, int(len(self.dataset) * 0.1))
        test_set = self.dataset[:test_size]
        val_result = self.validator.validate(potential, test_set)

        if val_result.status == "failed":
            self.logger.warning("Validation failed, but continuing for exploration...")
            # In a real system, we might stop or adjust strategy

        # 3. Exploration (MD)
        self.logger.info("Phase: Exploration")
        high_uncertainty_structures = self.dynamics_engine.run_exploration(potential)

        if not high_uncertainty_structures:
            self.logger.info("No high uncertainty structures found. Converged?")
            return CycleStatus.CONVERGED

        # Log max uncertainty found
        max_gamma = max(
            (
                s.uncertainty_state.gamma_max
                for s in high_uncertainty_structures
                if s.uncertainty_state and s.uncertainty_state.gamma_max
            ),
            default=0.0,
        )
        self.logger.info(
            f"Exploration found {len(high_uncertainty_structures)} structures. Max gamma: {max_gamma:.2f}"
        )

        # 4. Selection (Local Candidates & Active Set)
        self.logger.info("Phase: Selection")
        n_local = self.config.orchestrator.n_local_candidates

        # Batch generation of candidates (improved I/O efficiency)
        candidates = self.structure_generator.generate_batch_candidates(
            high_uncertainty_structures, n_candidates_per_seed=n_local
        )

        n_select = self.config.orchestrator.n_active_set_select
        active_set = self.trainer.select_active_set(candidates, n_select=n_select)

        # Filter candidates by ID (Optimized with Set)
        active_ids = set(active_set.structure_ids)
        selected_structures = [c for c in candidates if c.id in active_ids]

        # 5. Calculation (Oracle)
        self.logger.info(f"Phase: Calculation ({len(selected_structures)} structures)")
        new_data = self.oracle.compute_batch(selected_structures)
        self.dataset.extend(new_data)

        return CycleStatus.TRAINING
