import logging
from pathlib import Path
import uuid
import ase.io
import numpy as np
import itertools
from typing import Iterator, List

from mlip_autopipec.domain_models.config import Config
from mlip_autopipec.domain_models.workflow import WorkflowState
from mlip_autopipec.domain_models.dynamics import LammpsResult
from mlip_autopipec.domain_models.job import JobStatus
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.exploration import ExplorationTask
from mlip_autopipec.physics.structure_gen.generator import StructureGenFactory
from mlip_autopipec.physics.dynamics.lammps import LammpsRunner
from mlip_autopipec.physics.structure_gen.policy import AdaptivePolicy
from mlip_autopipec.physics.structure_gen.defects import DefectStrategy
from mlip_autopipec.physics.structure_gen.strain import StrainStrategy

logger = logging.getLogger("mlip_autopipec.phases.exploration")

class ExplorationPhase:
    def execute(self, state: WorkflowState, config: Config, work_dir: Path) -> LammpsResult:
        """
        Execute the Exploration Phase:
        1. Consult Adaptive Policy.
        2. Generate starting structure.
        3. Execute Task (MD or Static Generation).
        """
        logger.info("Running Exploration Phase with Adaptive Policy...")

        # 1. Consult Policy
        policy = AdaptivePolicy()
        task = policy.decide(state.generation, config)
        logger.info(f"Policy Decision: Method={task.method}, Modifiers={task.modifiers}")

        # 2. Generate Initial Structure (Base)
        gen_config = config.structure_gen
        generator = StructureGenFactory.get_generator(gen_config)
        base_structure = generator.generate(gen_config)
        logger.info(f"Generated base structure: {base_structure.get_chemical_formula()}")

        md_work_dir = work_dir / "md_run"
        md_work_dir.mkdir(exist_ok=True)

        # 3. Execute Task
        if task.method == "Static":
            return self._execute_static(task, base_structure, config, md_work_dir)
        else:
            return self._execute_dynamic(task, base_structure, state, config, md_work_dir)

    def _execute_static(
        self,
        task: ExplorationTask,
        base_structure: Structure,
        config: Config,
        work_dir: Path
    ) -> LammpsResult:
        """
        Executes static structure generation (Defects, Strain) and writes a fake trajectory.
        """
        # Helper to chain iterators
        iterators: List[Iterator[Structure]] = []

        # Apply strategies
        if "defect" in task.modifiers:
            logger.info("Applying DefectStrategy...")
            ds = DefectStrategy()
            iterators.append(ds.apply(base_structure, defect_type="vacancy"))
            iterators.append(ds.apply(base_structure, defect_type="interstitial"))
            if len(set(base_structure.symbols)) > 1:
                    iterators.append(ds.apply(base_structure, defect_type="antisite"))

        if "strain" in task.modifiers:
            logger.info("Applying StrainStrategy...")
            ss = StrainStrategy()
            iterators.append(ss.apply(base_structure, strain_type="uniaxial"))
            iterators.append(ss.apply(base_structure, strain_type="shear"))

        if "rattle" in task.modifiers:
            logger.info("Applying Rattle (via StrainStrategy)...")
            ss = StrainStrategy()
            iterators.append(ss.apply(base_structure, strain_type="rattle"))

        # Chain all iterators. This is LAZY evaluation.
        # `structure_stream` is a generator that yields structures one by one.
        # No list of structures is created in memory.
        structure_stream: Iterator[Structure]
        if not iterators:
            # If no modifiers or strategy failed, fallback to base
            # Use a generator expression to avoid creating a list in memory
            structure_stream = (base_structure for _ in range(1))
        else:
            structure_stream = itertools.chain(*iterators)

        # Write 'fake' trajectory
        # We use extxyz because ASE cannot write lammps-dump-text, and extxyz supports arrays
        traj_path = work_dir / "dump.extxyz"

        # Add high gamma to ensure selection
        threshold = config.orchestrator.uncertainty_threshold
        fake_gamma = threshold + 10.0

        # Stream writes to avoid OOM and excessive I/O operations
        # We track last structure to return it in result
        last_structure = base_structure
        count = 0

        # Use 'a' (append) mode but open once to minimize syscalls
        with open(traj_path, "w") as f:
            for s in structure_stream:
                atoms = s.to_ase()
                n_atoms = len(atoms)
                gamma_array = np.full(n_atoms, fake_gamma)
                atoms.new_array("c_pace_gamma", gamma_array) # type: ignore[no-untyped-call]

                # Write to file handle
                ase.io.write(f, atoms, format="extxyz") # type: ignore[no-untyped-call]

                last_structure = s
                count += 1

                # Explicitly delete to encourage GC
                del atoms
                del s

        if count == 0:
                ase.io.write(traj_path, base_structure.to_ase(), format="extxyz") # type: ignore[no-untyped-call]

        return LammpsResult(
            job_id="static_gen_" + str(uuid.uuid4())[:8],
            status=JobStatus.COMPLETED,
            work_dir=work_dir,
            duration_seconds=0.0,
            log_content="Generated by AdaptivePolicy (Static)",
            final_structure=last_structure,
            trajectory_path=traj_path,
            max_gamma=fake_gamma
        )

    def _execute_dynamic(
        self,
        task: ExplorationTask,
        base_structure: Structure,
        state: WorkflowState,
        config: Config,
        work_dir: Path
    ) -> LammpsResult:
        """
        Executes dynamic exploration (MD/MC) using LAMMPS.
        """
        md_config = config.md
        # Inject uncertainty threshold from Orchestrator config
        md_config.uncertainty_threshold = config.orchestrator.uncertainty_threshold

        # Override params from task if present
        if task.temperature:
            md_config.temperature = task.temperature
        if task.steps:
            md_config.n_steps = task.steps

        extra_commands = []
        if "swap" in task.modifiers:
            seed = config.potential.seed
            temp = md_config.temperature
            extra_commands.append(f"fix 2 all atom/swap 10 1 {seed} {temp}")

        runner = LammpsRunner(
            config=config.lammps,
            potential_config=config.potential,
            base_work_dir=work_dir
        )

        result = runner.run(
            base_structure,
            md_config,
            potential_path=state.latest_potential_path,
            extra_commands=extra_commands
        )

        return result
