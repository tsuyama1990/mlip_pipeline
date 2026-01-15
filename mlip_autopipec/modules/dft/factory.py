"""Module for the DFTFactory orchestrator."""

import copy
import logging
from pathlib import Path
from typing import Any

from ase import Atoms

from mlip_autopipec.config_schemas import DFTConfig
from mlip_autopipec.modules.dft.exceptions import DFTCalculationError
from mlip_autopipec.modules.dft.file_manager import QEFileManager
from mlip_autopipec.modules.dft.input_generator import QEInputGenerator
from mlip_autopipec.modules.dft.output_parser import QEOutputParser
from mlip_autopipec.modules.dft.process_runner import QEProcessRunner

logger = logging.getLogger(__name__)


def _apply_parameter_adjustment(
    config: DFTConfig, adjustment: dict[str, Any]
) -> DFTConfig:
    """Apply a single parameter adjustment to a deep copy of the config."""
    new_config = copy.deepcopy(config)
    for key, value in adjustment.items():
        parts = key.split(".")
        current_level: Any = new_config.input
        for part in parts[:-1]:
            current_level = getattr(current_level, part)
        setattr(current_level, parts[-1], value)
    return new_config


class DFTFactory:
    """Orchestrates a DFT calculation.

    This class acts as a high-level orchestrator, not a monolithic executor.
    It follows the principle of separation of concerns by delegating specific
    tasks to dedicated components:
    1.  `QEInputGenerator`: Generates the Quantum Espresso input file.
    2.  `QEProcessRunner`: Securely executes the `pw.x` command.
    3.  `QEOutputParser`: Parses the results from the output file.

    The factory's primary responsibility is to manage the retry loop, which
    catches common convergence failures and attempts to recover by modifying a
    copy of the DFT configuration for the next attempt.
    """

    def __init__(self, config: DFTConfig, base_work_dir: Path | None = None) -> None:
        """Initialize the DFTFactory.

        Args:
            config: The DFT-specific configuration object.
            base_work_dir: The base directory for temporary calculation folders.

        """
        self.config = config
        self.base_work_dir = base_work_dir
        self.input_generator = QEInputGenerator()
        self.process_runner = QEProcessRunner()
        self.output_parser = QEOutputParser()

    def run(self, atoms: Atoms) -> Atoms:
        """Run a DFT calculation with an automatic retry mechanism."""
        return self._run_calculation_with_retry(atoms)

    def _run_calculation_with_retry(self, atoms: Atoms) -> Atoms:
        """Execute the DFT calculation with a retry loop for robustness."""
        original_config = self.config
        max_retries = original_config.retry_strategy.max_retries

        for attempt in range(max_retries + 1):
            current_config = (
                _apply_parameter_adjustment(
                    original_config,
                    original_config.retry_strategy.parameter_adjustments[attempt - 1],
                )
                if attempt > 0
                else original_config
            )

            file_manager = QEFileManager()
            try:
                input_content = self.input_generator.generate(
                    atoms, config=current_config
                )
                file_manager.write_input(input_content)
                self.process_runner.execute(
                    file_manager.input_path,
                    file_manager.output_path,
                    config=current_config.executable,
                )
                results = self.output_parser.parse(file_manager.output_path)
                from ase.calculators.singlepoint import SinglePointCalculator

                atoms.calc = SinglePointCalculator(  # type: ignore[no-untyped-call]
                    atoms, **results
                )
                logger.info("DFT calculation succeeded on attempt %d.", attempt + 1)
                return atoms
            except DFTCalculationError as e:
                logger.warning(
                    "DFT attempt %d/%d failed.", attempt + 1, max_retries + 1
                )
                if attempt >= max_retries:
                    logger.error(
                        "DFT calculation failed after %d attempts.", max_retries + 1
                    )
                    raise e
            finally:
                file_manager.cleanup()
        raise DFTCalculationError("DFT calculation failed after all retries.")
