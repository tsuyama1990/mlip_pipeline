import logging
from pathlib import Path
from typing import List

import numpy as np
import ase.io

from mlip_autopipec.domain_models.config import Config, BulkStructureGenConfig
from mlip_autopipec.domain_models.job import JobStatus
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.workflow import (
    CandidateStatus,
    CandidateStructure,
    WorkflowPhase,
    WorkflowState,
)
from mlip_autopipec.orchestration.candidate_processing import CandidateManager
from mlip_autopipec.orchestration.state import StateManager
from mlip_autopipec.physics.dft.qe_runner import QERunner
from mlip_autopipec.physics.dynamics.lammps import LammpsRunner
from mlip_autopipec.physics.structure_gen.generator import StructureGenFactory
from mlip_autopipec.physics.training.dataset import DatasetManager
from mlip_autopipec.physics.training.pacemaker import PacemakerRunner
from mlip_autopipec.physics.validation.runner import ValidationRunner

logger = logging.getLogger("mlip_autopipec.orchestration")


class WorkflowManager:
    """
    The Brain of the Active Learning Loop.
    Manages state transitions and orchestrates workers.
    """

    def __init__(self, config: Config, work_dir: Path = Path(".")):
        self.config = config
        self.work_dir = work_dir
        self.state_manager = StateManager(work_dir)
        self.candidate_manager = CandidateManager()

        # Load or Init State
        loaded_state = self.state_manager.load()
        if loaded_state:
            self.state: WorkflowState = loaded_state
            logger.info(f"Resumed state: Gen {self.state.generation}, Phase {self.state.current_phase}")
        else:
            logger.info("No existing state found. Initializing new WorkflowState.")
            self.state = WorkflowState(
                project_name=config.project_name,
                dataset_path=work_dir / "data" / "accumulated.pckl.gzip",
                current_phase=WorkflowPhase.EXPLORATION,
                latest_potential_path=config.training.initial_potential if config.training else None
            )
            self.state_manager.save(self.state)

        # Setup Data Dir
        self.data_dir = work_dir / "data"
        self.data_dir.mkdir(exist_ok=True)
        self.dataset_manager = DatasetManager(self.data_dir)

    def run_loop(self):
        """
        Main execution loop.
        """
        logger.info("Starting Autonomous Active Learning Loop")

        while self.state.generation < self.config.orchestrator.max_iterations:
            should_continue = self.step()
            if not should_continue:
                break

        logger.info("Loop finished.")

    def step(self) -> bool:
        """
        Executes one step of the workflow based on the current phase.
        Returns True if the loop should continue, False if it should stop.
        """
        phase = self.state.current_phase
        gen_dir = self.work_dir / f"active_learning/iter_{self.state.generation:03d}"
        gen_dir.mkdir(parents=True, exist_ok=True)

        try:
            if phase == WorkflowPhase.EXPLORATION:
                halt_detected = self.explore(gen_dir)
                if halt_detected:
                    self.transition_to(WorkflowPhase.SELECTION)
                else:
                    logger.info("Exploration finished without halt. Convergence detected?")
                    return False

            elif phase == WorkflowPhase.SELECTION:
                md_dir = gen_dir / "md_run"
                candidates = self.select(md_dir)
                self.state.candidates = candidates
                self.transition_to(WorkflowPhase.CALCULATION)

            elif phase == WorkflowPhase.CALCULATION:
                success = self.calculate(gen_dir)
                if success:
                    self.transition_to(WorkflowPhase.TRAINING)
                else:
                    logger.error("Calculation phase failed.")
                    return False

            elif phase == WorkflowPhase.TRAINING:
                new_pot = self.train(gen_dir)
                self.state.latest_potential_path = new_pot
                self.transition_to(WorkflowPhase.VALIDATION)

            elif phase == WorkflowPhase.VALIDATION:
                self.validate(gen_dir)
                self.state.generation += 1
                self.transition_to(WorkflowPhase.EXPLORATION)

            return True

        except Exception as e:
            logger.exception(f"Error in phase {phase}: {e}")
            return False

    def transition_to(self, new_phase: WorkflowPhase):
        logger.info(f"Transition: {self.state.current_phase} -> {new_phase}")
        self.state.current_phase = new_phase
        self.state_manager.save(self.state)

    def explore(self, iter_dir: Path) -> bool:
        """
        Run MD. Return True if Halt Detected (High Uncertainty).
        """
        logger.info("Running Exploration...")

        gen_config = self.config.structure_gen
        generator = StructureGenFactory.get_generator(gen_config)
        structure = generator.generate(gen_config)

        md_config = self.config.md
        md_config.uncertainty_threshold = self.config.orchestrator.uncertainty_threshold

        runner = LammpsRunner(
            config=self.config.lammps,
            potential_config=self.config.potential,
            base_work_dir=iter_dir
        )

        result = runner.run(structure, md_config, potential_path=self.state.latest_potential_path)

        if result.max_gamma is not None:
             if result.max_gamma > self.config.orchestrator.uncertainty_threshold:
                 logger.info(f"Halt detected! Max Gamma: {result.max_gamma}")
                 return True

        return False

    def select(self, md_base_dir: Path) -> List[CandidateStructure]:
        """
        Scan MD trajectory for candidates.
        Applies Active Set Selection (D-Optimality) if candidates exceed max size.
        """
        logger.info("Selecting candidates...")
        job_dirs = sorted([d for d in md_base_dir.glob("job_*") if d.is_dir()])

        if not job_dirs:
            logger.warning("No MD job directories found.")
            return []

        latest_job = job_dirs[-1]
        traj_path = latest_job / "dump.lammpstrj"

        if not traj_path.exists():
            logger.warning("No trajectory found.")
            return []

        stride = self.config.orchestrator.trajectory_sampling_stride
        candidates = []
        threshold = self.config.orchestrator.uncertainty_threshold

        try:
             # First pass: collect all high-uncertainty candidates
             traj = ase.io.iread(traj_path, index=f"::{stride}", format="lammps-dump-text") # type: ignore[no-untyped-call]

             for i, atoms in enumerate(traj):
                 gammas = atoms.arrays.get('c_pace_gamma')
                 if gammas is not None and np.max(gammas) > threshold:
                     center_idx = int(np.argmax(gammas))

                     full_struct = Structure.from_ase(atoms)
                     cluster = self.candidate_manager.extract_cluster(
                         full_struct, center_idx, radius=self.config.potential.cutoff
                     )

                     cand = CandidateStructure(
                         structure=cluster,
                         origin=f"{latest_job.name}_frame_{i*stride}",
                         uncertainty_score=float(np.max(gammas)),
                         status=CandidateStatus.PENDING
                     )
                     candidates.append(cand)

        except Exception as e:
            logger.error(f"Error reading trajectory: {e}")
            return []

        # Limit size using D-Optimality (pace_activeset)
        max_size = self.config.orchestrator.max_active_set_size
        if len(candidates) > max_size and self.config.training:
            logger.info(f"Candidates ({len(candidates)}) > {max_size}. Running Active Set Selection.")

            # We need to leverage PacemakerRunner or call pace_activeset.
            # We will use a temporary directory for this operation.
            temp_dir = md_base_dir / "active_set_selection"
            temp_dir.mkdir(exist_ok=True)

            # Dump all candidates to extxyz
            all_cands_path = temp_dir / "candidates.extxyz"
            all_ase = [c.structure.to_ase() for c in candidates]
            ase.io.write(all_cands_path, all_ase) # type: ignore[no-untyped-call]

            # Instantiate PacemakerRunner solely for selection utility
            pm_runner = PacemakerRunner(
                work_dir=temp_dir,
                train_config=self.config.training,
                potential_config=self.config.potential
            )

            try:
                # pace_activeset writes to pckl.gzip
                selected_path = pm_runner.select_active_set(all_cands_path)

                # Load back the selected structures
                selected_structures = ase.io.read(selected_path, index=":") # type: ignore[no-untyped-call]

                if isinstance(selected_structures, list):
                    # We need to map back to candidates.
                    # Simple way: just keep the new structures as new candidates.
                    new_candidates = []
                    for s_ase in selected_structures:
                        # Re-wrap
                        s_struct = Structure.from_ase(s_ase)
                        new_candidates.append(CandidateStructure(
                            structure=s_struct,
                            origin="active_set_selection",
                            uncertainty_score=0.0, # Lost, or we need to preserve info in ase.info
                            status=CandidateStatus.PENDING
                        ))
                    candidates = new_candidates
                    logger.info(f"Reduced to {len(candidates)} candidates.")

            except Exception as e:
                logger.warning(f"Active set selection failed ({e}). Fallback to linear subsampling.")
                indices = np.linspace(0, len(candidates)-1, max_size, dtype=int)
                candidates = [candidates[i] for i in indices]

        elif len(candidates) > max_size:
             # Fallback if no training config or other issue
             indices = np.linspace(0, len(candidates)-1, max_size, dtype=int)
             candidates = [candidates[i] for i in indices]

        return candidates

    def calculate(self, iter_dir: Path) -> bool:
        """
        Run DFT for pending candidates.
        """
        logger.info(f"Calculating {len(self.state.candidates)} candidates...")

        if not self.config.dft:
            logger.error("DFT configuration missing.")
            return False

        dft_dir = iter_dir / "dft_calc"
        runner = QERunner(base_work_dir=dft_dir)

        processed_count = 0
        batch_size = self.config.orchestrator.dft_batch_size
        pending_writes = []

        for cand in self.state.candidates:
            if cand.status == CandidateStatus.PENDING:
                cand.status = CandidateStatus.CALCULATING
                self.state_manager.save(self.state)

                try:
                    # Embed
                    embedded = self.candidate_manager.embed_cluster(cand.structure)

                    # Run DFT
                    res = runner.run(embedded, self.config.dft)

                    if res.status == JobStatus.COMPLETED:
                        cand.status = CandidateStatus.DONE

                        # Update structure with DFT results
                        s_new = embedded.model_copy()
                        s_new.properties = {
                            'energy': res.energy,
                            'forces': res.forces,
                            'stress': res.stress
                        }

                        pending_writes.append(s_new)
                        processed_count += 1

                        if len(pending_writes) >= batch_size:
                            self.dataset_manager.convert(pending_writes, output_path=self.state.dataset_path, append=True)
                            pending_writes = []

                    else:
                        cand.status = CandidateStatus.FAILED
                        cand.error_message = res.log_content

                except Exception as e:
                    cand.status = CandidateStatus.FAILED
                    cand.error_message = str(e)

                self.state_manager.save(self.state)

        # Final flush
        if pending_writes:
            self.dataset_manager.convert(pending_writes, output_path=self.state.dataset_path, append=True)

        return processed_count > 0 or len(self.state.candidates) == 0

    def train(self, iter_dir: Path) -> Path:
        """
        Run Training.
        """
        logger.info("Training potential...")

        if not self.config.training:
             raise ValueError("Training configuration missing.")

        train_dir = iter_dir / "training"
        pacemaker = PacemakerRunner(
            work_dir=train_dir,
            train_config=self.config.training,
            potential_config=self.config.potential
        )

        # Set initial potential
        if self.state.latest_potential_path:
            pacemaker.train_config.initial_potential = self.state.latest_potential_path

        result = pacemaker.train(self.state.dataset_path)

        if result.status != JobStatus.COMPLETED:
            raise RuntimeError(f"Training failed: {result.log_content}")

        return result.potential_path

    def validate(self, iter_dir: Path) -> bool:
        """
        Run Validation.
        """
        logger.info("Validating potential...")

        if not self.state.latest_potential_path:
            return False

        # Generate bulk structure
        gen_config = self.config.structure_gen
        if isinstance(gen_config, BulkStructureGenConfig):
            gen_config = gen_config.model_copy(update={"rattle_stdev": 0.0})

        generator = StructureGenFactory.get_generator(gen_config)
        structure = generator.generate(gen_config)

        runner = ValidationRunner(
            val_config=self.config.validation,
            pot_config=self.config.potential,
            potential_path=self.state.latest_potential_path
        )

        result = runner.validate(structure)
        logger.info(f"Validation Result: {result.overall_status}")

        return True
