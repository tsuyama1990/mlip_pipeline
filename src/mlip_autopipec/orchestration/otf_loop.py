import logging
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.io import read, write

from mlip_autopipec.domain_models.dynamics import MDStatus
from mlip_autopipec.domain_models.exploration import ExplorationTask
from mlip_autopipec.domain_models.structures import CandidateStructure, StructureMetadata
from mlip_autopipec.physics.dynamics.lammps_runner import LammpsRunner
from mlip_autopipec.physics.structure_gen.strategies import RandomDisplacementGenerator

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
                    traj: Any = read(result.trajectory_path, index=":")

                    if not isinstance(traj, list):
                        traj = [traj]

                    if not traj:
                        logger.error("Trajectory empty")
                        return []

                    bad_structure = traj[-1]

                    # Save extracted structure
                    halt_id = f"halted_step_{result.halt_step}"
                    output_path = work_dir / f"{halt_id}.xyz"
                    write(output_path, bad_structure)

                    meta = StructureMetadata(
                        generation_method="md_halted",
                        parent_structure_id="seed",  # Tracking parent ID is generic here
                        source="otf_loop",
                    )
                    cand = CandidateStructure(
                        structure_path=output_path,
                        metadata=meta,
                    )
                    candidates.append(cand)

                    # Generate local candidates around halted structure
                    logger.info("Generating local candidates around halted structure.")

                    # Use task parameters or defaults
                    disp_range = task.parameters.get("local_displacement_range", 0.05)
                    local_count = task.parameters.get("local_sampling_count", 20)

                    generator = RandomDisplacementGenerator(
                        displacement_range=disp_range
                    )
                    local_candidates = generator.generate(
                        bad_structure, count=local_count
                    )

                    for i, local_atoms in enumerate(local_candidates):
                        local_path = work_dir / f"{halt_id}_local_{i}.xyz"
                        write(local_path, local_atoms)
                        local_meta = StructureMetadata(
                            generation_method="random_displacement_halt",
                            parent_structure_id=halt_id,
                            source="otf_loop",
                        )
                        candidates.append(
                            CandidateStructure(
                                structure_path=local_path, metadata=local_meta
                            )
                        )

                except Exception:
                    logger.exception("Failed to read trajectory or extract frame")

        elif result.status == MDStatus.COMPLETED:
            logger.info(
                "MD Completed without halt. No new candidates added from this run."
            )
            # In future cycles, we might add random sampling from successful trajectories.

        return candidates
