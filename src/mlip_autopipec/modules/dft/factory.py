"""Module for the DFTFactory orchestrator."""

import logging

from ase import Atoms

from mlip_autopipec.config.system import SystemConfig
from mlip_autopipec.modules.dft.exceptions import DFTCalculationError
from mlip_autopipec.modules.dft.file_manager import QEFileManager
from mlip_autopipec.modules.dft.input_generator import QEInputGenerator
from mlip_autopipec.modules.dft.output_parser import QEOutputParser
from mlip_autopipec.modules.dft.process_runner import QEProcessRunner

logger = logging.getLogger(__name__)


class DFTFactory:
    """A factory for running DFT calculations."""

    def __init__(self, config: SystemConfig) -> None:
        """Initialize the DFTFactory.

        Args:
            config: The fully-expanded system configuration object.

        """
        self.config = config
        self.input_generator = QEInputGenerator(config)
        self.process_runner = QEProcessRunner(config)
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
            atoms.calc.results = results

            return atoms
        except DFTCalculationError:
            logger.exception(
                "DFT calculation failed for structure: %s",
                atoms.get_chemical_formula(),  # type: ignore[no-untyped-call]
            )
            raise
        finally:
            file_manager.cleanup()
