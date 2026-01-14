"""Module for the DFTFactory orchestrator."""

import logging
from pathlib import Path

from ase import Atoms

from mlip_autopipec.config_schemas import DFTConfig
from mlip_autopipec.modules.dft.exceptions import DFTCalculationError
from mlip_autopipec.modules.dft.file_manager import QEFileManager
from mlip_autopipec.modules.dft.input_generator import QEInputGenerator
from mlip_autopipec.modules.dft.output_parser import QEOutputParser
from mlip_autopipec.modules.dft.process_runner import QEProcessRunner

logger = logging.getLogger(__name__)


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
        self.input_generator = QEInputGenerator(config)
        self.process_runner = QEProcessRunner(config.executable)
        self.output_parser = QEOutputParser()

    def run(self, atoms: Atoms) -> Atoms:
        """Run a DFT calculation.

        Args:
            atoms: The ASE `Atoms` object representing the structure.

        Returns:
            The input `Atoms` object with calculation results attached.

        """
        file_manager = QEFileManager()
        try:
            input_content = self.input_generator.generate(atoms)
            file_manager.write_input(input_content)

            self.process_runner.execute(
                file_manager.input_path, file_manager.output_path
            )

            results = self.output_parser.parse(file_manager.output_path)
            # Use a SinglePointCalculator to store the results, which is the
            # standard ASE practice for non-MD calculations.
            from ase.calculators.singlepoint import SinglePointCalculator

            atoms.calc = SinglePointCalculator(atoms, **results)

            return atoms
        except DFTCalculationError:
            logger.exception(
                "DFT calculation failed for structure: %s",
                atoms.get_chemical_formula(),  # type: ignore[no-untyped-call]
            )
            raise
        finally:
            file_manager.cleanup()
