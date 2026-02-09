import yaml
from pathlib import Path
from typing import Union
import logging
from mlip_autopipec.domain_models.config import (
    Config, GeneratorType, OracleType, TrainerType, DynamicsType, ValidatorType
)
from mlip_autopipec.core.logger import setup_logging
from mlip_autopipec.components.mock import (
    MockGenerator, MockOracle, MockTrainer, MockDynamics, MockValidator
)

# Placeholder for actual components (Cyclic Imports should be avoided)
# In future cycles, we'd import:
# from mlip_autopipec.components.generator.adaptive import AdaptiveGenerator
# from mlip_autopipec.components.oracle.dft import DFTOracle
# etc.


class Orchestrator:
    def __init__(self, config: Union[str, Path, Config]):
        """
        Initializes the Orchestrator.
        :param config: Path to config file or Config object.
        """
        if isinstance(config, (str, Path)):
            self.config = self._load_config(Path(config))
        else:
            self.config = config

        self.work_dir = Path(self.config.orchestrator.work_dir)
        self._initialize_workspace()

        setup_logging(self.work_dir)
        logging.info("Orchestrator initialized.")

        self._initialize_components()

    def _load_config(self, path: Path) -> Config:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return Config(**data)

    def _initialize_workspace(self):
        self.work_dir.mkdir(parents=True, exist_ok=True)
        (self.work_dir / "active_learning").mkdir(exist_ok=True)
        (self.work_dir / "potentials").mkdir(exist_ok=True)
        (self.work_dir / "data").mkdir(exist_ok=True)

    def _initialize_components(self):
        # Generator
        gen_conf = self.config.generator
        if gen_conf.type == GeneratorType.RANDOM:
            # self.generator = RandomGenerator(gen_conf, self.work_dir)
            self.generator = MockGenerator(gen_conf, self.work_dir) # Placeholder
        elif gen_conf.type == GeneratorType.ADAPTIVE:
            # self.generator = AdaptiveGenerator(gen_conf, self.work_dir)
            self.generator = MockGenerator(gen_conf, self.work_dir) # Placeholder
        else:
            raise NotImplementedError(f"Generator {gen_conf.type} not implemented")

        # Oracle
        oracle_conf = self.config.oracle
        if oracle_conf.type == OracleType.MOCK:
            self.oracle = MockOracle(oracle_conf, self.work_dir)
        elif oracle_conf.type == OracleType.DFT:
            # self.oracle = DFTOracle(oracle_conf, self.work_dir)
            self.oracle = MockOracle(oracle_conf, self.work_dir) # Placeholder
        else:
            raise NotImplementedError(f"Oracle {oracle_conf.type} not implemented")

        # Trainer
        trainer_conf = self.config.trainer
        if trainer_conf.type == TrainerType.MOCK:
            self.trainer = MockTrainer(trainer_conf, self.work_dir)
        elif trainer_conf.type == TrainerType.PACEMAKER:
            # self.trainer = PacemakerTrainer(trainer_conf, self.work_dir)
            self.trainer = MockTrainer(trainer_conf, self.work_dir) # Placeholder
        else:
            raise NotImplementedError(f"Trainer {trainer_conf.type} not implemented")

        # Dynamics
        dyn_conf = self.config.dynamics
        if dyn_conf.type == DynamicsType.MOCK:
            self.dynamics = MockDynamics(dyn_conf, self.work_dir)
        elif dyn_conf.type == DynamicsType.LAMMPS:
            # self.dynamics = LAMMPSDynamics(dyn_conf, self.work_dir)
            self.dynamics = MockDynamics(dyn_conf, self.work_dir) # Placeholder
        elif dyn_conf.type == DynamicsType.EON:
             # self.dynamics = EONDynamics(dyn_conf, self.work_dir)
             self.dynamics = MockDynamics(dyn_conf, self.work_dir) # Placeholder
        else:
            raise NotImplementedError(f"Dynamics {dyn_conf.type} not implemented")

        # Validator
        val_conf = self.config.validator
        if val_conf.type == ValidatorType.MOCK:
            self.validator = MockValidator(val_conf, self.work_dir)
        elif val_conf.type == ValidatorType.STANDARD:
            # self.validator = StandardValidator(val_conf, self.work_dir)
            self.validator = MockValidator(val_conf, self.work_dir) # Placeholder
        else:
            raise NotImplementedError(f"Validator {val_conf.type} not implemented")

    def run_cycle(self):
        """
        Runs the active learning cycle.
        """
        logging.info("Starting Active Learning Cycle...")
        # Minimal implementation for Cycle 01
        # In future cycles, this will be the loop logic

        # Example flow:
        # structures = self.generator.generate(10)
        # computed = self.oracle.compute_batch(structures)
        # potential = self.trainer.train(computed)
        # validated = self.validator.validate(potential)

        logging.info("Cycle completed (Mock).")
