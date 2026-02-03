import logging
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.io import read, write

from mlip_autopipec.domain_models.dynamics import MDStatus
from mlip_autopipec.domain_models.exploration import ExplorationTask
from mlip_autopipec.domain_models.structures import CandidateStructure, StructureMetadata
from mlip_autopipec.physics.dynamics.lammps_runner import LammpsRunner

logger = logging.getLogger(__name__)


class OTFLoop:
    def __init__(self, runner: LammpsRunner) -> None:
        self.runner = runner

    def execute_task(
        self,
        task: ExplorationTask,
        seed: Atoms,
        potential_path: Path | None,
        work_dir: Path,
    ) -> list[CandidateStructure]:
        """
        Executes an MD exploration task with OTF monitoring.
        """
        # Merge parameters
        params = task.parameters.copy()

        # Run MD
        result = self.runner.run(seed, potential_path, work_dir, params)

        candidates: list[CandidateStructure] = []

        if result.status == MDStatus.HALTED:
            logger.info(f"MD Halted at step {result.halt_step}. Extracting structure.")
            if result.trajectory_path and result.trajectory_path.exists():
                try:
                    # Read trajectory
                    # We assume the last frame is the one of interest (high uncertainty)
                    traj: Any = read(result.trajectory_path, index=":")  # type: ignore[no-untyped-call]

                    if not isinstance(traj, list):
                        traj = [traj]

                    if not traj:
                        logger.error("Trajectory empty")
                        return []

                    bad_structure = traj[-1]

                    # Save extracted structure
                    output_path = work_dir / f"halted_step_{result.halt_step}.xyz"
                    write(output_path, bad_structure)  # type: ignore[no-untyped-call]

                    meta = StructureMetadata(
                        generation_method="md_halted",
                        parent_structure_id="seed",  # Tracking parent ID is generic here
                    )
                    cand = CandidateStructure(
                        structure_path=output_path,
                        metadata=meta,
                    )
                    candidates.append(cand)

                except Exception:
                    logger.exception("Failed to read trajectory or extract frame")

        elif result.status == MDStatus.COMPLETED:
            logger.info("MD Completed without halt. No new candidates added from this run.")
            # In future cycles, we might add random sampling from successful trajectories.

        return candidates
