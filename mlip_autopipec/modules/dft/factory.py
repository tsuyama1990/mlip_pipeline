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
from mlip_autopipec.utils.resilience import retry

logger = logging.getLogger(__name__)


class DFTFactory:
    """Orchestrates a DFT calculation.

    This class acts as a high-level orchestrator, not a monolithic executor.
    It follows the principle of separation of concerns by delegating specific
    tasks to dedicated components:
    1.  `QEInputGenerator`: Generates the Quantum Espresso input file.
    2.  `QEProcessRunner`: Securely executes the `pw.x` command.
    3.  `QEOutputParser`: Parses the results from the output file.

    Resilience is handled by the `@retry` decorator, which is configured
    via the `retry_strategy` section of the DFT configuration. This keeps
    the core execution logic clean and separates the retry mechanism into
    a reusable utility.
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

        # The retry decorator is applied in the `_run_calculation` method
        # to keep this public method clean. The number of retries is sourced
        # directly from the validated configuration schema.
        @retry(
            max_retries=self.config.retry_strategy.max_retries,
            exceptions=(DFTCalculationError,),
        )
        def _run_dft_with_retry() -> Atoms:
            return self._run_single_calculation(atoms)

        return _run_dft_with_retry()

    def _run_single_calculation(self, atoms: Atoms) -> Atoms:
        """Execute a single DFT calculation attempt.

        This method encapsulates the logic for a single run, which can be
        decorated for retries. It manages the lifecycle of the temporary
        files required for the calculation.

        Args:
            atoms: The atomic structure to calculate.

        Returns:
            The Atoms object with the calculated results attached.

        Raises:
            DFTCalculationError: If any step of the DFT calculation fails.

        """
        file_manager = QEFileManager()
        try:
            input_content = self.input_generator.generate(atoms, config=self.config)
            file_manager.write_input(input_content)
            self.process_runner.execute(
                file_manager.input_path,
                file_manager.output_path,
                config=self.config.executable,
            )
            results = self.output_parser.parse(file_manager.output_path)
            from ase.calculators.singlepoint import SinglePointCalculator

            atoms.calc = SinglePointCalculator(atoms, **results)  # type: ignore[no-untyped-call]
            logger.info("DFT calculation succeeded.")
            return atoms
        except DFTCalculationError:
            # Re-raising the exception allows the @retry decorator to catch it.
            # Logging of the failure and retry attempts is handled by the decorator.
            raise
        finally:
            file_manager.cleanup()
