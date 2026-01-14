import tempfile
from pathlib import Path

from mlip_autopipec.schemas.dft import DFTInput, DFTOutput
from mlip_autopipec.utils.logging import get_logger

from .qe_input_generator import QEInputGenerator
from .qe_output_parser import QEOutputParser
from .qe_process_runner import QEProcessRunner

logger = get_logger(__name__)


class QERunner:
    """
    A wrapper for Quantum Espresso to run DFT calculations.

    This class orchestrates the process of running a DFT calculation by
    generating an input file, running the QE process, and parsing the
    output. It also includes a recovery mechanism to handle convergence
    failures.
    """

    def __init__(self, max_retries: int = 3, keep_temp_dir: bool = False) -> None:
        """
        Initializes the QERunner.

        Args:
            max_retries: The maximum number of times to retry a failed
                calculation.
            keep_temp_dir: Whether to keep the temporary directory
                used for the calculation. Useful for debugging.
        """
        self.max_retries = max_retries
        self.keep_temp_dir = keep_temp_dir

    def run(self, dft_input: DFTInput) -> DFTOutput:
        """
        Runs a DFT calculation and returns the output.

        Args:
            dft_input: The input for the DFT calculation.

        Returns:
            The output of the DFT calculation.

        Raises:
            RuntimeError: If the calculation fails after multiple retries.
        """
        for attempt in range(self.max_retries):
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                if self.keep_temp_dir:
                    tmpdir_path = Path(tempfile.mkdtemp())

                input_generator = QEInputGenerator(dft_input)
                input_generator.write_input(tmpdir_path)

                process_runner = QEProcessRunner(tmpdir_path)
                process = process_runner.run()

                if process.returncode == 0 and "!" in process.stdout:
                    output_parser = QEOutputParser(process.stdout)
                    return output_parser.parse()

                logger.warning(f"QE calculation failed on attempt {attempt + 1}. Recovering.")
                dft_input = self._recover(dft_input)

        error_message = "Quantum Espresso failed after multiple retries."
        logger.error(error_message)
        raise RuntimeError(error_message)

    def _recover(self, dft_input: DFTInput) -> DFTInput:
        """Modifies DFT parameters to attempt recovery from a failed run."""
        new_params = dft_input.dft_params.model_copy(deep=True)
        new_params.mixing_beta = 0.3
        return DFTInput(atoms=dft_input.atoms, dft_params=new_params)
