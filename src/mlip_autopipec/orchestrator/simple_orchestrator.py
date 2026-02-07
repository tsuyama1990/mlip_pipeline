import logging
from collections.abc import Iterator

from ase.io import read

from mlip_autopipec.domain_models import (
    ExplorationResult,
    ExplorationStatus,
    GlobalConfig,
    Structure,
)
from mlip_autopipec.factory import (
    create_dynamics,
    create_generator,
    create_oracle,
    create_selector,
    create_trainer,
    create_validator,
)
from mlip_autopipec.utils import setup_logging

logger = logging.getLogger(__name__)


class SimpleOrchestrator:
    def __init__(self, config: GlobalConfig) -> None:
        self.config = config
        self.workdir = config.workdir
        self.workdir.mkdir(parents=True, exist_ok=True)
        setup_logging(self.workdir)

        self.oracle = create_oracle(config.oracle)
        self.trainer = create_trainer(config.trainer)
        self.dynamics = create_dynamics(config.dynamics)
        self.generator = create_generator(config.generator)
        self.validator = create_validator(config.validator)
        self.selector = create_selector(config.selector)

        self.dataset_path = self.workdir / "dataset.jsonl"

        # Initialize dataset from initial structure
        if not self.dataset_path.exists():
            self._initialize_dataset()

    def _initialize_dataset(self) -> None:
        logger.info(f"Initializing dataset from {self.config.initial_structure_path}")
        if not self.config.initial_structure_path.exists():
            logger.warning(
                f"Initial structure path {self.config.initial_structure_path} does not exist. Starting with empty dataset."
            )
            self.dataset_path.touch()
            return

        # ASE read can return Atom or list of Atoms
        # We handle untyped call explicitly
        structures_ase = read(str(self.config.initial_structure_path), index=":")
        if not isinstance(structures_ase, list):
            structures_ase = [structures_ase]

        with self.dataset_path.open("w") as f:
            for atoms in structures_ase:
                s = Structure(atoms=atoms)
                f.write(s.model_dump_json() + "\n")

    def _load_dataset(self) -> Iterator[Structure]:
        with self.dataset_path.open("r") as f:
            for line in f:
                if line.strip():
                    yield Structure.model_validate_json(line)

    def _append_to_dataset(self, structures: Iterator[Structure]) -> None:
        count = 0
        with self.dataset_path.open("a") as f:
            for s in structures:
                f.write(s.model_dump_json() + "\n")
                count += 1
        logger.info(f"Appended {count} structures to dataset.")

    def run(self) -> None:
        logger.info("Starting Orchestrator loop...")

        for cycle in range(self.config.max_cycles):
            logger.info(f"=== Cycle {cycle + 1}/{self.config.max_cycles} ===")
            try:
                self._run_cycle(cycle)
            except Exception as e:
                logger.error(f"Error in cycle {cycle}: {e}", exc_info=True)

    def _run_cycle(self, cycle: int) -> None:
        # 1. Train
        logger.info("Training potential...")
        dataset_iter = self._load_dataset()
        cycle_workdir = self.workdir / f"cycle_{cycle}"
        potential = self.trainer.train(dataset_iter, cycle_workdir)
        logger.info(f"Potential trained at {potential.path}")

        # 2. Dynamics / Exploration
        logger.info("Running dynamics...")
        last_structure = self._get_last_structure()

        if last_structure is None:
            last_structure = self._generate_initial_structure()
            if last_structure is None:
                return

        exploration_result = self.dynamics.run(potential, last_structure, cycle_workdir)

        if exploration_result.status == ExplorationStatus.CONVERGED:
            logger.info("Dynamics converged.")
        else:
            self._handle_halted_dynamics(exploration_result)

        # 6. Validation
        logger.info("Validating...")
        # Using empty test set for mock
        validation_res = self.validator.validate(potential, [], cycle_workdir)
        logger.info(
            f"Validation passed: {validation_res.passed}, metrics: {validation_res.metrics}"
        )

    def _get_last_structure(self) -> Structure | None:
        last_structure: Structure | None = None
        for s in self._load_dataset():
            last_structure = s
        return last_structure

    def _generate_initial_structure(self) -> Structure | None:
        logger.info("Dataset is empty! Generating initial structures...")
        initial_structs = self.generator.generate(n=1)
        gen_list = list(initial_structs)
        if not gen_list:
            logger.error("Generator returned no structures. Aborting cycle.")
            return None

        # Label and append
        labeled_gen = list(self.oracle.compute(gen_list))
        self._append_to_dataset(iter(labeled_gen))

        if labeled_gen:
            return labeled_gen[0]
        return None

    def _handle_halted_dynamics(self, exploration_result: ExplorationResult) -> None:
        logger.info("Dynamics halted/uncertain. Selecting new structures...")
        # 3. Selection
        candidates: list[Structure] = []
        if exploration_result.trajectory_path.exists():
            # Read trajectory
            traj = read(str(exploration_result.trajectory_path), index=":")
            if isinstance(traj, list):
                candidates = [Structure(atoms=at) for at in traj]
            else:
                candidates = [Structure(atoms=traj)]
        else:
            candidates = [exploration_result.final_structure]

        selected_iter = self.selector.select(candidates, n=1)

        # 4. Labeling (Oracle)
        labeled_iter = self.oracle.compute(selected_iter)

        # 5. Update Dataset
        self._append_to_dataset(labeled_iter)
