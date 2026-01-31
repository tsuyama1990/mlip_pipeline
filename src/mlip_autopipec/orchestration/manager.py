import logging
import shutil
from pathlib import Path
from typing import List, Iterator, Optional

import numpy as np
import ase.io
from ase import Atoms

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
from mlip_autopipec.infrastructure.io import load_structures

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
                if new_pot is not None:
                    self.state.latest_potential_path = new_pot
                    self.transition_to(WorkflowPhase.VALIDATION)
                else:
                    logger.error("Training returned no potential path.")
                    return False

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
        Scan MD trajectory for candidates using streaming/chunking to avoid OOM.
        Uses pace_activeset (D-Optimality) if available.
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
        threshold = self.config.orchestrator.uncertainty_threshold

        # Helper to stream candidates
        def stream_candidates() -> Iterator[CandidateStructure]:
            traj = ase.io.iread(traj_path, index=f"::{stride}", format="lammps-dump-text") # type: ignore[no-untyped-call]
            for i, atoms in enumerate(traj):
                if not isinstance(atoms, Atoms):
                    continue

                gammas = atoms.arrays.get('c_pace_gamma')
                if gammas is not None and np.max(gammas) > threshold:
                    center_idx = int(np.argmax(gammas))

                    full_struct = Structure.from_ase(atoms)
                    cluster = self.candidate_manager.extract_cluster(
                        full_struct, center_idx, radius=self.config.potential.cutoff
                    )

                    yield CandidateStructure(
                        structure=cluster,
                        origin=f"{latest_job.name}_frame_{i*stride}",
                        uncertainty_score=float(np.max(gammas)),
                        status=CandidateStatus.PENDING
                    )

        # Temporary path for all candidates if we use active set selection
        temp_dir = md_base_dir / "active_set_selection"
        temp_dir.mkdir(exist_ok=True)
        all_cands_path = temp_dir / "candidates.extxyz"

        # Stream writes to avoid OOM
        count = 0

        if self.config.training and self.config.orchestrator.active_set_optimization:
             # Streaming Reservoir Sampling
             target_max = self.config.orchestrator.max_active_set_size
             reservoir_cap = target_max * 10
             reservoir: List[Atoms] = []

             for cand in stream_candidates():
                 if len(reservoir) < reservoir_cap:
                     atoms = cand.structure.to_ase()
                     # Store metadata separately or in info
                     atoms.info['origin'] = cand.origin
                     atoms.info['gamma'] = cand.uncertainty_score
                     reservoir.append(atoms)
                 else:
                     j = np.random.randint(0, count + 1)
                     if j < reservoir_cap:
                         atoms = cand.structure.to_ase()
                         atoms.info['origin'] = cand.origin
                         atoms.info['gamma'] = cand.uncertainty_score
                         reservoir[j] = atoms
                 count += 1

             if not reservoir:
                 shutil.rmtree(temp_dir)
                 return []

             # Write reservoir to file
             logger.info(f"Writing {len(reservoir)} pre-screened candidates to disk for Active Set Selection.")
             ase.io.write(all_cands_path, reservoir) # type: ignore[no-untyped-call]

             # Run pace_activeset
             pm_runner = PacemakerRunner(
                work_dir=temp_dir,
                train_config=self.config.training,
                potential_config=self.config.potential
             )

             try:
                 selected_path = pm_runner.select_active_set(all_cands_path)

                 # Read back selected
                 final_candidates = []
                 for s_struct in load_structures(selected_path):

                     origin = str(s_struct.properties.get('origin', 'active_set_selection'))
                     gamma = float(s_struct.properties.get('gamma', 0.0))

                     final_candidates.append(CandidateStructure(
                         structure=s_struct,
                         origin=origin,
                         uncertainty_score=gamma,
                         status=CandidateStatus.PENDING
                     ))

                 logger.info(f"Selected {len(final_candidates)} candidates.")
                 shutil.rmtree(temp_dir) # Cleanup
                 return final_candidates

             except Exception as e:
                 logger.warning(f"Active set selection failed: {e}. Fallback to iterative subsampling.")

                 target = self.config.orchestrator.max_active_set_size

                 # Iterative Subsampling (Reservoir Sampling-like)
                 stride_fallback = max(1, int(np.ceil(count / target))) if target > 0 else 1

                 final_candidates = []
                 for i, s_struct in enumerate(load_structures(all_cands_path)):
                     if i % stride_fallback == 0:
                         origin = str(s_struct.properties.get('origin', ''))
                         gamma = float(s_struct.properties.get('gamma', 0.0))

                         final_candidates.append(CandidateStructure(
                             structure=s_struct,
                             origin=origin,
                             uncertainty_score=gamma,
                             status=CandidateStatus.PENDING
                         ))
                         # Safety break if we slightly exceed due to integer math
                         if len(final_candidates) >= target:
                             break

                 shutil.rmtree(temp_dir) # Cleanup
                 return final_candidates

        else:
            # If no active set opt, use simple iterative subsampling on stream
            max_size = self.config.orchestrator.max_active_set_size
            reservoir_result: List[CandidateStructure] = []

            for i, cand in enumerate(stream_candidates()):
                if len(reservoir_result) < max_size:
                    reservoir_result.append(cand)
                else:
                    j = np.random.randint(0, i + 1)
                    if j < max_size:
                        reservoir_result[j] = cand

            # Cleanup if temp dir was created (it wasn't in this branch, but good practice)
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

            return reservoir_result

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

                        # Force Masking (SPEC 3.2)
                        # We zero out forces for atoms outside cutoff (buffer region)
                        # Candidates carry 'cluster_dist' array from extraction
                        if "cluster_dist" in cand.structure.arrays:
                            dists = cand.structure.arrays["cluster_dist"]
                            cutoff = self.config.potential.cutoff
                            mask_indices = np.where(dists > cutoff)[0]
                            if len(mask_indices) > 0:
                                res.forces[mask_indices] = 0.0
                                logger.debug(f"Masked forces for {len(mask_indices)} buffer atoms.")

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

    def train(self, iter_dir: Path) -> Optional[Path]:
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
