import logging
from pathlib import Path
from typing import Any

from ase.io import read, write

from mlip_autopipec.config import Config
from mlip_autopipec.domain_models.exploration import (
    AKMCTask,
    ExplorationTask,
    MDTask,
    StaticTask,
)
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

    def explore(
        self, potential_path: Path | None, work_dir: Path
    ) -> list[CandidateStructure]:
        # 1. Load Seed
        seed_path = self.config.training.dataset_path
        if not seed_path.exists():
            return []

        atoms_or_list: Any = read(seed_path, index=-1)
        seed_atoms = (
            atoms_or_list[0] if isinstance(atoms_or_list, list) else atoms_or_list
        )

        # 2. Decide Strategy
        uncertainty = 1.0 if potential_path is None else 0.5
        tasks = self.policy.decide_strategy(seed_atoms, uncertainty)

        candidates = []

        # 3. Execute Tasks
        for i, task in enumerate(tasks):
            candidates.extend(
                self._execute_task(i, task, seed_atoms, potential_path, work_dir)
            )

        return candidates

    def _execute_task(
        self,
        index: int,
        task: ExplorationTask,
        seed_atoms: Any,
        potential_path: Path | None,
        work_dir: Path,
    ) -> list[CandidateStructure]:
        if isinstance(task, StaticTask):
            return self._run_static_task(index, task, seed_atoms, work_dir)

        if isinstance(task, MDTask):
            return self._run_md_task(index, task, seed_atoms, potential_path, work_dir)

        if isinstance(task, AKMCTask):
            return self._run_akmc_task(
                index, task, seed_atoms, potential_path, work_dir
            )

        return []

    def _run_static_task(
        self, index: int, task: StaticTask, seed_atoms: Any, work_dir: Path
    ) -> list[CandidateStructure]:
        candidates = []
        new_structs = []
        gen: StructureGenerator

        if "strain" in task.modifiers:
            # Type safe access
            rng = task.parameters.strain_range
            gen = StrainGenerator(strain_range=rng)
            count = 20
            new_structs = gen.generate(seed_atoms, count=count)

        elif "defect" in task.modifiers:
            # Type safe access
            dtype = task.parameters.defect_type
            gen = DefectGenerator(defect_type=dtype)
            count = 1
            new_structs = gen.generate(seed_atoms, count=count)

        for j, at in enumerate(new_structs):
            fname = f"candidate_t{index}_{j}.xyz"
            fpath = work_dir / fname
            write(fpath, at)

            meta = StructureMetadata(generation_method=f"static_{task.modifiers[0]}")
            cand = CandidateStructure(
                structure_path=fpath,
                metadata=meta,
            )
            candidates.append(cand)
        return candidates

    def _run_md_task(
        self,
        index: int,
        task: MDTask,
        seed_atoms: Any,
        potential_path: Path | None,
        work_dir: Path,
    ) -> list[CandidateStructure]:
        if self.otf_loop:
            logger.info(f"Executing MD Task {index}")
            task_dir = work_dir / f"task_{index}_md"
            return self.otf_loop.execute_task(
                task, seed_atoms, potential_path, task_dir
            )
        logger.warning("MD task requested but Lammps not configured.")
        return []

    def _run_akmc_task(
        self,
        index: int,
        task: AKMCTask,
        seed_atoms: Any,
        potential_path: Path | None,
        work_dir: Path,
    ) -> list[CandidateStructure]:
        logger.info(f"Executing AKMC Task {index}")
        task_dir = work_dir / f"task_{index}_akmc"
        wrapper = EonWrapper(self.config)

        if potential_path is None:
            logger.warning("AKMC requested but no potential available.")
            return []

        wrapper.run_akmc(potential_path, seed_atoms, task_dir)
        return self._collect_eon_results(task_dir)

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
                            structure_path=geom_path, metadata=meta
                        )
                        candidates.append(cand)
        return candidates
