import tempfile
from pathlib import Path

from mlip_autopipec.schemas.dft import DFTInput, DFTOutput
from mlip_autopipec.utils.logging import get_logger

from .qe_input_generator import QEInputGenerator
from .qe_output_parser import QEOutputParser
from .qe_process_runner import QEProcessRunner

logger = get_logger(__name__)


def run_qe_calculation(
    dft_input: DFTInput, max_retries: int = 3, keep_temp_dir: bool = False
) -> DFTOutput:
    """
    Runs a DFT calculation and returns the output.

    Args:
        dft_input: The input for the DFT calculation.
        max_retries: The maximum number of times to retry a failed
            calculation.
        keep_temp_dir: Whether to keep the temporary directory
            used for the calculation. Useful for debugging.

    Returns:
        The output of the DFT calculation.

    Raises:
        RuntimeError: If the calculation fails after multiple retries.
    """
    for attempt in range(max_retries):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            if keep_temp_dir:
                tmpdir_path = Path(tempfile.mkdtemp())

            input_generator = QEInputGenerator(dft_input)
            input_generator.write_input(tmpdir_path)

            process_runner = QEProcessRunner(tmpdir_path)
            process = process_runner.run()

            if process.returncode == 0 and "!" in process.stdout:
                output_parser = QEOutputParser(process.stdout)
                return output_parser.parse()

            logger.warning(f"QE calculation failed on attempt {attempt + 1}. Recovering.")
            dft_input = _recover(dft_input)

    error_message = "Quantum Espresso failed after multiple retries."
    logger.exception(error_message)
    raise RuntimeError(error_message)


def _recover(dft_input: DFTInput) -> DFTInput:
    """Modifies DFT parameters to attempt recovery from a failed run."""
    new_params = dft_input.dft_params.model_copy(deep=True)
    new_params.mixing_beta = 0.3
    return DFTInput(atoms=dft_input.atoms, dft_params=new_params)
