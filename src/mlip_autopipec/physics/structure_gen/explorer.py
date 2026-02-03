import logging
from pathlib import Path
from typing import Any

from ase.io import read, write

from mlip_autopipec.config import Config
from mlip_autopipec.domain_models.exploration import ExplorationMethod
from mlip_autopipec.domain_models.structures import CandidateStructure, StructureMetadata
from mlip_autopipec.orchestration.otf_loop import OTFLoop
from mlip_autopipec.physics.dynamics.eon_wrapper import EonWrapper
from mlip_autopipec.physics.structure_gen.generator import StructureGenerator
from mlip_autopipec.physics.structure_gen.policy import AdaptivePolicy
from mlip_autopipec.physics.structure_gen.strategies import DefectGenerator, StrainGenerator

logger = logging.getLogger(__name__)


class AdaptiveExplorer:
    def __init__(self, config: Config, otf_loop: OTFLoop | None = None) -> None:
        self.config = config
        self.policy = AdaptivePolicy()
        self.otf_loop = otf_loop

    def explore(self, potential_path: Path | None, work_dir: Path) -> list[CandidateStructure]:
        # 1. Load Seed
        seed_path = self.config.training.dataset_path
        if not seed_path.exists():
            return []

        atoms_or_list: Any = read(seed_path, index=-1)
        seed_atoms = atoms_or_list[0] if isinstance(atoms_or_list, list) else atoms_or_list

        # 2. Decide Strategy
        uncertainty = 1.0 if potential_path is None else 0.5
        tasks = self.policy.decide_strategy(seed_atoms, uncertainty)

        candidates = []

        # 3. Execute Tasks
        for i, task in enumerate(tasks):
            if task.method == ExplorationMethod.STATIC:
                new_structs = []
                gen: StructureGenerator

                if "strain" in task.modifiers:
                    rng = task.parameters.get("strain_range", 0.05)
                    gen = StrainGenerator(strain_range=rng)
                    count = 20
                    new_structs = gen.generate(seed_atoms, count=count)

                elif "defect" in task.modifiers:
                    dtype = task.parameters.get("defect_type", "vacancy")
                    gen = DefectGenerator(defect_type=dtype)
                    count = 1
                    new_structs = gen.generate(seed_atoms, count=count)

                for j, at in enumerate(new_structs):
                    fname = f"candidate_t{i}_{j}.xyz"
                    fpath = work_dir / fname
                    write(fpath, at)

                    meta = StructureMetadata(generation_method=f"static_{task.modifiers[0]}")
                    cand = CandidateStructure(
                        structure_path=fpath,
                        metadata=meta,
                    )
                    candidates.append(cand)

            elif task.method == ExplorationMethod.MD:
                if self.otf_loop:
                    logger.info(f"Executing MD Task {i}")
                    task_dir = work_dir / f"task_{i}_md"
                    new_cands = self.otf_loop.execute_task(
                        task, seed_atoms, potential_path, task_dir
                    )
                    candidates.extend(new_cands)
                else:
                    logger.warning("MD task requested but Lammps not configured.")

            elif task.method == ExplorationMethod.AKMC:
                logger.info(f"Executing AKMC Task {i}")
                task_dir = work_dir / f"task_{i}_akmc"
                wrapper = EonWrapper(self.config)

                if potential_path is None:
                    logger.warning("AKMC requested but no potential available.")
                    continue

                wrapper.run_akmc(potential_path, seed_atoms, task_dir)
                candidates.extend(self._collect_eon_results(task_dir))

        return candidates

    def _collect_eon_results(self, task_dir: Path) -> list[CandidateStructure]:
        candidates = []
        states_dir = task_dir / "states"
        if states_dir.exists():
            for state_path in states_dir.iterdir():
                if state_path.is_dir():
                    # EON typically stores 'geometry.con' or 'pos.con' in state dirs
                    geom_path = state_path / "geometry.con"
                    if not geom_path.exists():
                        geom_path = state_path / "pos.con"

                    if geom_path.exists():
                        meta = StructureMetadata(generation_method="akmc_state")
                        cand = CandidateStructure(
                            structure_path=geom_path,
                            metadata=meta
                        )
                        candidates.append(cand)
        return candidates
