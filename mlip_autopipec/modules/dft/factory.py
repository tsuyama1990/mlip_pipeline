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
    """A factory for running DFT calculations."""

    def __init__(self, config: DFTConfig, base_work_dir: Path | None = None) -> None:
        """Initialize the DFTFactory.

        Args:
            config: The DFT-specific configuration object.
            base_work_dir: The base directory for temporary calculation folders.

        """
        self.config = config
        self.base_work_dir = base_work_dir
        self.output_parser = QEOutputParser()

    def run(self, atoms: Atoms) -> Atoms:
        """Run a DFT calculation with an automatic retry mechanism.

        This method attempts to run the DFT calculation. If it fails, it will
        sequentially apply parameter adjustments defined in the `retry_strategy`
        and retry the calculation up to `max_retries` times.

        Args:
            atoms: The ASE `Atoms` object representing the structure.

        Returns:
            The input `Atoms` object with calculation results attached.

        Raises:
            DFTCalculationError: If the calculation fails after all retries.

        """
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
                input_generator = QEInputGenerator(current_config)
                process_runner = QEProcessRunner(current_config.executable)

                input_content = input_generator.generate(atoms)
                file_manager.write_input(input_content)

                process_runner.execute(
                    file_manager.input_path, file_manager.output_path
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
                        "DFT calculation failed after %d attempts for structure: %s",
                        max_retries + 1,
                        atoms.get_chemical_formula(),  # type: ignore[no-untyped-call]
                    )
                    raise e
            finally:
                file_manager.cleanup()

        # This part should be unreachable, but it's here for type safety
        raise DFTCalculationError("DFT calculation failed after all retries.")
