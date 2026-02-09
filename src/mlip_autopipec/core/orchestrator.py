import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from mlip_autopipec.core.candidate_generator import generate_local_candidates
from mlip_autopipec.core.dataset import Dataset
from mlip_autopipec.core.state import StateManager
from mlip_autopipec.domain_models.config import GlobalConfig
from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.results import ValidationMetrics
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.factory import ComponentFactory
from mlip_autopipec.utils.security import validate_safe_path

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Main pipeline orchestrator.

    Architecture Note:
    This implementation follows a "Batch Active Learning" strategy (Cycle 1 -> N),
    which is more robust for automated pipelines than the "On-the-Fly (OTF) Halt & Resume"
    strategy described in the original Spec (Section 3.4).

    Instead of halting the MD simulation mid-run (which requires complex process control
    and state management), we run Exploration (Dynamics) to completion, collecting
    uncertain structures into a batch. This batch is then Labeled (Oracle),
    added to the Dataset, and used for Training a new Potential for the next cycle.
    """

    def __init__(self, config: GlobalConfig) -> None:
        self.config = config

        # Resolve paths from config
        self.dataset_path = config.workdir / config.orchestrator.dataset_filename
        self.state_path = config.workdir / config.orchestrator.state_filename

        # Initialize Core Components
        self.state_manager = StateManager(self.state_path)
        self.dataset = Dataset(self.dataset_path, root_dir=config.workdir)

        # Instantiate components via Factory (Dependency Injection)
        self.generator = ComponentFactory.get_generator(config.components.generator)
        self.oracle = ComponentFactory.get_oracle(config.components.oracle)
        self.trainer = ComponentFactory.get_trainer(config.components.trainer)
        self.dynamics = ComponentFactory.get_dynamics(config.components.dynamics)
        self.validator = ComponentFactory.get_validator(config.components.validator)

        self.current_potential: Potential | None = None
        self.halt_count = 0

    def __repr__(self) -> str:
        return f"<Orchestrator(workdir={self.config.workdir}, cycle={self.state_manager.state.current_cycle})>"

    def __str__(self) -> str:
        return f"Orchestrator(cycle={self.state_manager.state.current_cycle})"

    def run(self) -> None:
        """Run the active learning loop until max_cycles is reached."""
        logger.info("Starting Orchestrator")
        self.state_manager.update_status("RUNNING")

        try:
            while self.state_manager.state.current_cycle < self.config.max_cycles:
                try:
                    self._run_cycle()
                except StopIteration:
                    logger.info("Pipeline Converged.")
                    break

                self.state_manager.update_cycle(self.state_manager.state.current_cycle + 1)

            if self.state_manager.state.status != "CONVERGED":
                logger.info("Max cycles reached. Stopping.")
                self.state_manager.update_status("STOPPED")

        except Exception:
            logger.exception("Orchestrator failed")
            self.state_manager.update_status("ERROR")
            raise

    def _run_cycle(self) -> None:
        """
        Execute one full active learning cycle:
        Exploration -> Labeling -> Dataset Update -> Training -> Validation.
        """
        cycle = self.state_manager.state.current_cycle + 1
        logger.info(f"=== Starting Cycle {cycle:02d} ===")

        # Reset halt count
        self.halt_count = 0

        # Use configured cycle directory pattern
        cycle_dir_name = self.config.orchestrator.cycle_dir_pattern.format(cycle=cycle)
        cycle_dir = self.config.workdir / cycle_dir_name
        cycle_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Exploration
        structures = self._explore(cycle, cycle_dir)

        # Step 2: Labeling
        logger.info(f"[Cycle {cycle}] Labeling: Computing DFT properties")
        labeled_structures = self.oracle.compute(structures)

        # Step 3: Dataset Update
        logger.info(f"[Cycle {cycle}] Dataset: Appending new data")
        self.dataset.append(labeled_structures)

        logger.info(f"[Cycle {cycle}] Exploration Stats: {self.halt_count} halts encountered.")

        # Step 4: Training
        logger.info(f"[Cycle {cycle}] Training: Fitting new potential")
        self.current_potential = self.trainer.train(
            dataset=self.dataset, workdir=cycle_dir, previous_potential=self.current_potential
        )
        logger.info(f"Potential trained: {self.current_potential.path}")

        # Step 5: Check Convergence & Validation
        self._validate_and_check_convergence(cycle, cycle_dir)

        logger.info(f"=== Cycle {cycle:02d} Completed ===")

    def _explore(self, cycle: int, cycle_dir: Path) -> Iterator[Structure]:
        """Step 1: Exploration phase."""
        n_structures = self.config.components.generator.n_structures

        if cycle == 1:
            logger.info(f"[Cycle {cycle}] Exploration: Generating initial structures (Cold Start)")
            return self.generator.generate(n_structures=n_structures)

        logger.info(f"[Cycle {cycle}] Exploration: Running Dynamics with previous potential")
        self._ensure_potential(cycle)

        # Generate start structures
        start_structures_iter = self.generator.generate(n_structures=n_structures)

        physics_baseline_dict = None
        if self.config.physics_baseline:
            physics_baseline_dict = self.config.physics_baseline.model_dump()

        raw_structures = self.dynamics.explore(
            self.current_potential,
            start_structures_iter,
            workdir=cycle_dir,
            physics_baseline=physics_baseline_dict,
        )
        return self._enhance_structures(raw_structures)

    def _ensure_potential(self, cycle: int) -> None:
        """Load potential from previous cycle if missing."""
        if self.current_potential is None:
            prev_cycle = cycle - 1
            prev_cycle_dir_name = self.config.orchestrator.cycle_dir_pattern.format(
                cycle=prev_cycle
            )
            prev_pot_path = (
                self.config.workdir
                / prev_cycle_dir_name
                / self.config.orchestrator.potential_filename
            )

            try:
                safe_prev_pot_path = validate_safe_path(prev_pot_path, must_exist=True)
                self.current_potential = Potential(path=safe_prev_pot_path)
                logger.info(f"Loaded potential from {safe_prev_pot_path}")
            except Exception as e:
                msg = f"Cycle {cycle}: Failed to load potential from Cycle {prev_cycle} at {prev_pot_path}: {e}"
                raise RuntimeError(msg) from e

    def _validate_and_check_convergence(self, cycle: int, cycle_dir: Path) -> None:
        """Step 5: Run validation and check for convergence."""
        metrics: ValidationMetrics | None = None

        if cycle > 1 and self.halt_count == 0:
            logger.info(f"[Cycle {cycle}] Convergence check passed (no halts). Running Validation.")
            metrics = self.validator.validate(self.current_potential)
            logger.info(f"Validation metrics: {metrics}")

            self.current_potential.metrics.update(metrics.model_dump())

            if metrics.passed:
                self.state_manager.update_status("CONVERGED")
                self._save_metrics(cycle_dir, metrics)
                logger.info(f"=== Cycle {cycle:02d} Completed (Converged) ===")
                # Use a specific exception or message for clarity
                msg = "Converged"
                raise StopIteration(msg)

            logger.info("Validation failed. Adding failure cases to dataset.")
            if metrics.failed_structures:
                logger.info(f"Labeling {len(metrics.failed_structures)} failed structures.")
                labeled_failures = self.oracle.compute(iter(metrics.failed_structures))
                self.dataset.append(labeled_failures)
        else:
            logger.info(f"[Cycle {cycle}] Skipping Validation (Halts: {self.halt_count})")

        self._save_metrics(cycle_dir, metrics)

    def _enhance_structures(self, structures: Iterator[Structure]) -> Iterator[Structure]:
        """
        Enhance stream of structures with local candidates if they are halted.
        """
        for s in structures:
            provenance = s.tags.get("provenance", "")
            if "halt" in str(provenance).lower():
                self.halt_count += 1
                logger.info("Generating local candidates for halted structure")
                candidates = generate_local_candidates(s, n_candidates=20)
                selected = self.trainer.select_active_set(candidates, limit=6)
                yield from selected
            else:
                yield s

    def _save_metrics(self, cycle_dir: Path, metrics: ValidationMetrics | None) -> None:
        """Save cycle metrics to JSON."""
        data: dict[str, Any] = {
            "cycle": self.state_manager.state.current_cycle + 1,
            "dataset_size": len(self.dataset),
            "halt_count": self.halt_count,
            "validation_run": metrics is not None,
        }

        if metrics:
            data.update(metrics.model_dump())

        metrics_file = cycle_dir / "metrics.json"
        metrics_file.write_text(json.dumps(data, indent=2, default=str))
