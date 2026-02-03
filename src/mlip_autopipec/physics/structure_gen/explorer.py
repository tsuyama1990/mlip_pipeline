from pathlib import Path

from ase.io import read, write

from mlip_autopipec.config import Config
from mlip_autopipec.domain_models.exploration import ExplorationMethod
from mlip_autopipec.domain_models.structures import CandidateStructure, StructureMetadata
from mlip_autopipec.physics.structure_gen.policy import AdaptivePolicy
from mlip_autopipec.physics.structure_gen.strategies import DefectGenerator, StrainGenerator


class AdaptiveExplorer:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.policy = AdaptivePolicy()

    def explore(
        self, potential_path: Path | None, work_dir: Path
    ) -> list[CandidateStructure]:
        # 1. Load Seed
        seed_path = self.config.training.dataset_path
        if not seed_path.exists():
            # For UAT, we assume it exists. If not, return empty list.
            return []

        # Read the last structure as seed (most relevant?)
        # For simplicity, read last one (index -1)
        # Using type ignore because read return type is dynamic
        seed_atoms = read(seed_path, index=-1)  # type: ignore[no-untyped-call]

        # 2. Decide Strategy
        # Uncertainty: We don't have uncertainty calculation yet.
        uncertainty = 1.0 if potential_path is None else 0.5

        tasks = self.policy.decide_strategy(seed_atoms, uncertainty)

        candidates = []

        # 3. Execute Tasks
        for i, task in enumerate(tasks):
            if task.method == ExplorationMethod.STATIC:
                new_structs = []
                # Check modifiers
                if "strain" in task.modifiers:
                    rng = task.parameters.get("strain_range", 0.05)
                    gen = StrainGenerator(strain_range=rng)
                    # How many? 20 according to UAT 03-01
                    count = 20
                    new_structs = gen.generate(seed_atoms, count=count)

                elif "defect" in task.modifiers:
                    dtype = task.parameters.get("defect_type", "vacancy")
                    gen = DefectGenerator(defect_type=dtype)
                    count = 1  # Just 1 defect structure per task?
                    new_structs = gen.generate(seed_atoms, count=count)

                # Save and wrap
                for j, at in enumerate(new_structs):
                    # Filename: candidate_task_i_j.xyz
                    fname = f"candidate_t{i}_{j}.xyz"
                    fpath = work_dir / fname
                    write(fpath, at)  # type: ignore[no-untyped-call]

                    meta = StructureMetadata(generation_method=f"static_{task.modifiers[0]}")
                    cand = CandidateStructure(
                        structure_path=fpath,
                        metadata=meta,
                    )
                    candidates.append(cand)

        return candidates
