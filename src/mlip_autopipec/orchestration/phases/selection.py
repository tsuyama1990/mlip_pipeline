import logging
import shutil
import numpy as np
import ase.io
from pathlib import Path
from typing import List, Iterator

from ase import Atoms
from mlip_autopipec.domain_models.config import Config
from mlip_autopipec.domain_models.workflow import WorkflowState, CandidateStructure, CandidateStatus
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.orchestration.candidate_processing import CandidateManager
from mlip_autopipec.infrastructure.io import load_structures
from mlip_autopipec.physics.training.pacemaker import PacemakerRunner

logger = logging.getLogger("mlip_autopipec.phases.selection")

class SelectionPhase:
    def execute(self, state: WorkflowState, config: Config, work_dir: Path) -> List[CandidateStructure]:
        logger.info("Selecting candidates...")

        md_base_dir = work_dir / "md_run"
        job_dirs = sorted([d for d in md_base_dir.glob("job_*") if d.is_dir()])

        if not job_dirs:
            logger.warning("No MD job directories found.")
            return []

        latest_job = job_dirs[-1]
        traj_path = latest_job / "dump.lammpstrj"

        if not traj_path.exists():
            logger.warning("No trajectory found.")
            return []

        stride = config.orchestrator.trajectory_sampling_stride
        threshold = config.orchestrator.uncertainty_threshold
        candidate_manager = CandidateManager()

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
                    cluster = candidate_manager.extract_cluster(
                        full_struct, center_idx, radius=config.potential.cutoff
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

        if config.training and config.orchestrator.active_set_optimization:
             # Streaming Reservoir Sampling
             target_max = config.orchestrator.max_active_set_size
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

             # Write reservoir to file in chunks
             logger.info(f"Writing {len(reservoir)} pre-screened candidates to disk for Active Set Selection.")

             chunk_size = 100
             for i in range(0, len(reservoir), chunk_size):
                 chunk = reservoir[i : i + chunk_size]
                 # Use append=True for write if supported or just 'a' mode
                 # ase.io.write supports append=True for some formats like extxyz
                 if i == 0:
                     ase.io.write(all_cands_path, chunk, format='extxyz') # type: ignore[no-untyped-call]
                 else:
                     ase.io.write(all_cands_path, chunk, format='extxyz', append=True) # type: ignore[no-untyped-call]

             # Run pace_activeset
             pm_runner = PacemakerRunner(
                work_dir=temp_dir,
                train_config=config.training,
                potential_config=config.potential
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

                 target = config.orchestrator.max_active_set_size

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
            max_size = config.orchestrator.max_active_set_size
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
