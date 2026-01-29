"""UAT Verification Script for Cycle 02."""

import logging
import sys
from pathlib import Path
import numpy as np

# Ensure src is in path
sys.path.append(str(Path.cwd() / "src"))

from mlip_autopipec.domain_models.config import Config, ExplorationConfig
from mlip_autopipec.domain_models.workflow import WorkflowState, WorkflowPhase
from mlip_autopipec.orchestration.phases.exploration import ExplorationPhase

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UAT-02")

def uat_02_01_cold_start() -> bool:
    logger.info("Running UAT-02-01: Cold Start Generation")

    # 1. Configure
    config = Config(
        project_name="UAT_02",
        structure_gen=ExplorationConfig(
            strategy="template",
            composition="Si",
            num_candidates=5,
            rattle_amplitude=0.1
        )
    )

    state = WorkflowState(current_phase=WorkflowPhase.EXPLORATION)

    # 2. Run
    phase = ExplorationPhase()
    phase.execute(state, config)

    # 3. Inspect
    if len(state.candidates) != 5:
        logger.error(f"Failed: Expected 5 candidates, got {len(state.candidates)}")
        return False

    for i, cand in enumerate(state.candidates):
        if "Si" not in cand.formatted_formula:
             logger.error(f"Failed: Candidate {i} formula mismatch: {cand.formatted_formula}")
             return False

    logger.info("UAT-02-01 Passed")
    return True

def uat_02_02_validity() -> bool:
    logger.info("Running UAT-02-02: Structure Validity")

    # 1. Configure with large rattle
    config = Config(
        project_name="UAT_02",
        structure_gen=ExplorationConfig(
            strategy="template",
            composition="Al",
            num_candidates=2,
            rattle_amplitude=0.5
        )
    )

    state = WorkflowState(current_phase=WorkflowPhase.EXPLORATION)
    phase = ExplorationPhase()
    phase.execute(state, config)

    # 4. Verify no atoms overlap too much
    # Simple check: min distance > 0.5 A
    for cand in state.candidates:
        atoms = cand.to_ase()
        # ASE get_all_distances
        dist_matrix = atoms.get_all_distances(mic=True) # type: ignore[no-untyped-call]
        # Mask diagonal
        np.fill_diagonal(dist_matrix, 10.0)
        min_dist = np.min(dist_matrix)

        if min_dist < 0.5:
             logger.error(f"Failed: Atomic overlap detected. Min dist: {min_dist}")
             return False

    logger.info("UAT-02-02 Passed")
    return True

if __name__ == "__main__":
    success = True
    success &= uat_02_01_cold_start()
    success &= uat_02_02_validity()

    if success:
        logger.info("All UAT tests passed.")
        sys.exit(0)
    else:
        logger.error("Some UAT tests failed.")
        sys.exit(1)
