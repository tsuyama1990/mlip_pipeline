import logging
import typer
import yaml
from typing import Optional, List, Union, Any
from pathlib import Path
from ase import Atoms
from ase.io import read

from mlip_autopipec.config.config_model import GlobalConfig
from mlip_autopipec.domain_models.structure import Dataset, Structure
from mlip_autopipec.domain_models.potential import Potential, ExplorationResult
from mlip_autopipec.domain_models.validation import ValidationResult

from mlip_autopipec.interfaces.oracle import BaseOracle
from mlip_autopipec.interfaces.trainer import BaseTrainer
from mlip_autopipec.interfaces.explorer import BaseExplorer
from mlip_autopipec.interfaces.validator import BaseValidator

# Import Mocks
from mlip_autopipec.infrastructure.mocks import MockOracle, MockTrainer, MockExplorer, MockValidator
from mlip_autopipec.utils.logging import setup_logging

app = typer.Typer()
logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(
        self,
        config: GlobalConfig,
        oracle: Optional[BaseOracle] = None,
        trainer: Optional[BaseTrainer] = None,
        explorer: Optional[BaseExplorer] = None,
        validator: Optional[BaseValidator] = None
    ) -> None:
        self.config = config
        self.work_dir = config.work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Dependency Injection or Factory
        self.oracle = oracle or self._create_oracle()
        self.trainer = trainer or self._create_trainer()
        self.explorer = explorer or self._create_explorer()
        self.validator = validator or self._create_validator()

        self.current_potential: Optional[Potential] = None
        self.cycle_count = 0

    def _create_oracle(self) -> BaseOracle:
        if self.config.oracle.type == "mock":
            return MockOracle(self.config.oracle, self.work_dir)
        msg = f"Unknown Oracle type: {self.config.oracle.type}"
        raise ValueError(msg)

    def _create_trainer(self) -> BaseTrainer:
        if self.config.trainer.type == "mock":
            return MockTrainer(self.config.trainer, self.work_dir)
        msg = f"Unknown Trainer type: {self.config.trainer.type}"
        raise ValueError(msg)

    def _create_explorer(self) -> BaseExplorer:
        if self.config.explorer.type == "mock":
            return MockExplorer(self.config.explorer, self.work_dir)
        msg = f"Unknown Explorer type: {self.config.explorer.type}"
        raise ValueError(msg)

    def _create_validator(self) -> BaseValidator:
        # Currently Validator is not in config, assuming Mock for Cycle 01
        return MockValidator()

    def _extract_structures(self, result: ExplorationResult) -> List[Structure]:
        """
        Extracts structures from the dump file based on halted frames.
        """
        if not result.dump_file.exists():
            logger.warning(f"Dump file {result.dump_file} does not exist.")
            return []

        structures: List[Structure] = []
        try:
            # ase.io.read can return Atoms or list of Atoms
            trajectory: Union[Atoms, List[Atoms]] = read(result.dump_file, index=":") # type: ignore[no-untyped-call]
            traj_list: List[Atoms] = trajectory if isinstance(trajectory, list) else [trajectory]

            for i in result.high_gamma_frames:
                if 0 <= i < len(traj_list):
                    atoms = traj_list[i]
                    structures.append(Structure(atoms=atoms, metadata={"source": f"cycle_{self.cycle_count}_frame_{i}"}))
                else:
                    logger.warning(f"Frame {i} out of bounds for trajectory of length {len(traj_list)}")
        except Exception:
            logger.exception("Failed to read dump file")

        return structures

    def run(self) -> None:
        logger.info("Orchestrator started.")

        # Initial Potential
        self.current_potential = Potential(path=self.work_dir / "initial.yace", name="initial")

        for i in range(1, self.config.max_cycles + 1):
            self.cycle_count = i
            logger.info(f"--- Starting Cycle {i} ---")

            # 1. Explore
            exploration_result = self.explorer.explore(self.current_potential)

            if exploration_result.halted:
                logger.info("Exploration halted. Extracting candidates.")

                # 2. Extract Candidates
                candidates = self._extract_structures(exploration_result)
                if not candidates:
                    logger.warning("No candidates extracted. Skipping cycle.")
                    continue

                dataset = Dataset(structures=candidates, name=f"candidates_cycle_{i}")
                logger.info(f"Extracted {len(dataset)} candidates.")

                # 3. Label (Oracle)
                labeled_dataset = self.oracle.compute(dataset)

                # 4. Train
                self.current_potential = self.trainer.train(labeled_dataset, self.current_potential)
                logger.info(f"New potential trained: {self.current_potential.path}")

                # 5. Validate
                validation_res = self.validator.validate(self.current_potential)
                logger.info(f"Validation result: {validation_res}")

            else:
                logger.info("Exploration finished without halt. Continuing to next cycle.")

        logger.info("Orchestrator finished.")

@app.command()
def main(
    config_path: Path = typer.Argument(..., help="Path to the configuration YAML file"), # noqa: B008
    log_level: str = typer.Option("INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)") # noqa: B008
) -> None:
    """
    Run the MLIP Active Learning Pipeline.
    """
    setup_logging(log_level)

    if not config_path.exists():
        typer.echo(f"Error: Config file {config_path} not found.", err=True)
        raise typer.Exit(code=1)

    try:
        config_data = yaml.safe_load(config_path.read_text())
        config = GlobalConfig(**config_data)

        orchestrator = Orchestrator(config)
        orchestrator.run()

    except Exception as e:
        logger.exception("An error occurred during execution.")
        raise typer.Exit(code=1) from e

if __name__ == "__main__":
    app()
