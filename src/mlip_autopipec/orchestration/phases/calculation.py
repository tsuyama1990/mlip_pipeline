import logging
import numpy as np
from pathlib import Path
from typing import Optional, Callable

from mlip_autopipec.domain_models.config import Config
from mlip_autopipec.domain_models.workflow import WorkflowState, CandidateStatus
from mlip_autopipec.domain_models.job import JobStatus
from mlip_autopipec.orchestration.candidate_processing import CandidateManager
from mlip_autopipec.physics.dft.qe_runner import QERunner
from mlip_autopipec.physics.training.dataset import DatasetManager

logger = logging.getLogger("mlip_autopipec.phases.calculation")

class CalculationPhase:
    def execute(self, state: WorkflowState, config: Config, work_dir: Path, save_state_callback: Optional[Callable[[], None]] = None) -> bool:
        logger.info(f"Calculating {len(state.candidates)} candidates...")

        if not config.dft:
            logger.error("DFT configuration missing.")
            return False

        dft_dir = work_dir / "dft_calc"
        runner = QERunner(base_work_dir=dft_dir)
        candidate_manager = CandidateManager()

        # Dataset Manager uses orchestrator data dir (usually passed in state or config)
        # In Manager it was self.data_dir. In State we have dataset_path.
        # We need the directory of dataset_path.
        dataset_manager = DatasetManager(work_dir=state.dataset_path.parent)

        processed_count = 0
        batch_size = config.orchestrator.dft_batch_size
        pending_writes = []

        for cand in state.candidates:
            if cand.status == CandidateStatus.PENDING:
                cand.status = CandidateStatus.CALCULATING
                if save_state_callback:
                    save_state_callback()

                try:
                    # Embed
                    embedded = candidate_manager.embed_cluster(cand.structure)

                    # Run DFT
                    res = runner.run(embedded, config.dft)

                    if res.status == JobStatus.COMPLETED:
                        cand.status = CandidateStatus.DONE

                        # Update structure with DFT results
                        s_new = embedded.model_copy()

                        # Force Masking (SPEC 3.2)
                        # We zero out forces for atoms outside cutoff (buffer region)
                        # Candidates carry 'cluster_dist' array from extraction
                        if "cluster_dist" in cand.structure.arrays:
                            dists = cand.structure.arrays["cluster_dist"]
                            cutoff = config.potential.cutoff
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
                            dataset_manager.convert(pending_writes, output_path=state.dataset_path, append=True)
                            pending_writes = []

                    else:
                        cand.status = CandidateStatus.FAILED
                        cand.error_message = res.log_content

                except Exception as e:
                    cand.status = CandidateStatus.FAILED
                    cand.error_message = str(e)
                    logger.error(f"DFT Error: {e}")

                if save_state_callback:
                    save_state_callback()

        # Final flush
        if pending_writes:
            dataset_manager.convert(pending_writes, output_path=state.dataset_path, append=True)

        return processed_count > 0 or len(state.candidates) == 0
