import logging
import random
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.io import read, write

from mlip_autopipec.domain_models.dynamics import MDStatus
from mlip_autopipec.domain_models.exploration import MDTask
from mlip_autopipec.domain_models.structures import CandidateStructure, StructureMetadata
from mlip_autopipec.physics.dynamics.lammps_runner import LammpsRunner
from mlip_autopipec.physics.structure_gen.embedding import extract_periodic_box
from mlip_autopipec.physics.structure_gen.strategies import RandomDisplacementGenerator

logger = logging.getLogger(__name__)


class OTFLoop:
    def __init__(self, runner: LammpsRunner) -> None:
        self.runner = runner

    def execute_task(
        self,
        task: MDTask,
        seed: Atoms,
        potential_path: Path | None,
        work_dir: Path,
    ) -> list[CandidateStructure]:
        """
        Executes an MD exploration task with OTF monitoring.
        """
        # Convert MDParameters to dict for LammpsRunner
        params = task.parameters.model_dump()

        # Run MD
        result = self.runner.run(seed, potential_path, work_dir, params)

        candidates: list[CandidateStructure] = []

        if result.status == MDStatus.HALTED:
            logger.info(f"MD Halted at step {result.halt_step}. Initiating recovery.")
            if result.trajectory_path and result.trajectory_path.exists():
                try:
                    # Read trajectory
                    traj: Any = read(result.trajectory_path, index=":")
                    if not isinstance(traj, list):
                        traj = [traj]

                    if not traj:
                        logger.error("Trajectory empty")
                        return []

                    # 1. Extract Halted Structure
                    bad_structure = traj[-1]

                    # 2. Embed (Periodic Box)
                    if result.halt_atom_id is not None:
                        center_index = result.halt_atom_id
                    else:
                        # Fallback heuristic if ID extraction failed
                        center_index = random.randint(0, len(bad_structure) - 1)  # noqa: S311

                    # Use typed parameters
                    cutoff = task.parameters.embedding_cutoff

                    embedded_anchor = extract_periodic_box(bad_structure, center_index, cutoff)

                    # Save Anchor
                    anchor_path = work_dir / f"halted_anchor_step_{result.halt_step}.xyz"
                    write(anchor_path, embedded_anchor)

                    candidates.append(
                        CandidateStructure(
                            structure_path=anchor_path,
                            metadata=StructureMetadata(
                                source="md_halt",
                                generation_method="embedding",
                                parent_structure_id="seed_md",
                                uncertainty=1.0,  # High uncertainty triggered halt
                            ),
                        )
                    )

                    # 3. Generate Local Candidates
                    disp_range = task.parameters.local_displacement_range
                    count = task.parameters.local_sampling_count

                    generator = RandomDisplacementGenerator(displacement_range=disp_range)
                    local_candidates = generator.generate(embedded_anchor, count=count)

                    for i, cand_atoms in enumerate(local_candidates):
                        cand_path = work_dir / f"halted_local_{i}_step_{result.halt_step}.xyz"
                        write(cand_path, cand_atoms)

                        candidates.append(
                            CandidateStructure(
                                structure_path=cand_path,
                                metadata=StructureMetadata(
                                    source="md_halt_local_search",
                                    generation_method="random_displacement",
                                    parent_structure_id="halted_anchor",
                                ),
                            )
                        )

                    logger.info(f"Generated {len(candidates)} candidates from halt event.")

                except Exception:
                    logger.exception("Failed to recover from halt event")

        elif result.status == MDStatus.COMPLETED:
            logger.info("MD Completed without halt. No new candidates added from this run.")

        return candidates
