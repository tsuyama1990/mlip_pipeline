"""
This module contains the components for creating, managing, and running DFT
calculations, forming a key part of the data generation pipeline.
"""
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ase.atoms import Atoms

from mlip_autopipec.config.models import (
    CutoffConfig,
    DFTInputParameters,
    DFTJob,
    DFTResult,
    MagnetismConfig,
    Pseudopotentials,
    SmearingConfig,
)
from mlip_autopipec.exceptions import DFTCalculationError
from mlip_autopipec.utils.resilience import retry

from .dft_handlers.input_generator import QEInputGenerator
from .dft_handlers.output_parser import QEOutputParser
from .dft_handlers.process_runner import QEProcessRunner

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
SSSP_DATA_PATH = DATA_DIR / "sssp_p_data.json"
MAGNETIC_ELEMENTS = {"Fe", "Co", "Ni", "Cr", "Mn"}


from mlip_autopipec.utils.data_loader import load_sssp_data


class DFTHeuristics:
    """
    A dedicated class for generating sensible DFT parameters based on heuristics.

    This class encapsulates the domain-specific knowledge required to choose
    parameters like k-points, cutoffs, and magnetism settings for a given
    atomic structure. It is designed to be a stateless utility class.

    NOTE: The methods in this class are designed to be generic and data-driven,
    avoiding repeated logic for different chemical elements. The parameters
    are derived from loops and lookups into the SSSP data file.

    Args:
        sssp_data_path: The path to the JSON file containing the SSSP data.
    """

    def __init__(self, sssp_data_path: Path):
        self._sssp_data = load_sssp_data(sssp_data_path)

    def get_heuristic_parameters(self, atoms: Atoms) -> DFTInputParameters:
        """
        Determines a set of reasonable DFT parameters for a given structure.
        """
        elements = set(atoms.get_chemical_symbols())
        return DFTInputParameters(
            pseudopotentials=self._get_pseudopotentials(elements),
            cutoffs=self._get_heuristic_cutoffs(elements),
            k_points=self._get_heuristic_k_points(atoms),
            smearing=SmearingConfig(),
            magnetism=self._get_heuristic_magnetism(elements),
        )

    def _get_heuristic_cutoffs(self, elements: set[str]) -> CutoffConfig:
        """Determines cutoffs from SSSP data for the present elements."""
        max_wfc = max(self._sssp_data[el]["cutoff_wfc"] for el in elements)
        max_rho = max(self._sssp_data[el]["cutoff_rho"] for el in elements)
        return CutoffConfig(wavefunction=max_wfc, density=max_rho)

    def _get_heuristic_k_points(self, atoms: Atoms) -> tuple[int, int, int]:
        """Calculates a k-point grid based on a target density."""
        k_density = 6.0
        k_points = [
            max(1, int(k_density / length) + 1) if length > 0 else 1
            for length in atoms.cell.lengths()
        ]
        return (k_points[0], k_points[1], k_points[2])

    def _get_heuristic_magnetism(self, elements: set[str]) -> MagnetismConfig | None:
        """Enables magnetism if any magnetic elements are present."""
        if any(el in MAGNETIC_ELEMENTS for el in elements):
            mag_dict = dict.fromkeys(elements, 0.5)
            return MagnetismConfig(starting_magnetization=mag_dict)
        return None

    def _get_pseudopotentials(self, elements: set[str]) -> Pseudopotentials:
        """Retrieves pseudopotential filenames for the given elements."""
        pseudos = {el: self._sssp_data[el]["filename"] for el in elements}
        return Pseudopotentials.model_validate(pseudos)


class DFTJobFactory:
    """
    Orchestrates the creation of `DFTJob` objects.

    This class acts as a high-level factory. It utilizes a `DFTHeuristics`
    instance to determine the appropriate parameters and then constructs a
    validated `DFTJob` object, which is ready for execution by a `DFTRunner`.
    This separation of concerns makes the process more modular and testable.

    Args:
        heuristics: An instance of `DFTHeuristics` to use for parameter
                    generation.
    """

    def __init__(self, heuristics: DFTHeuristics):
        self.heuristics = heuristics

    def create_job(self, atoms: Atoms) -> DFTJob:
        """
        Creates a DFTJob with heuristic parameters for a given atomic structure.
        """
        params = self.heuristics.get_heuristic_parameters(atoms)
        return DFTJob(atoms=atoms, params=params)


def dft_retry_handler(exception: Exception, kwargs: dict[str, Any]) -> dict[str, Any] | None:
    """
    Handles specific DFT convergence errors by suggesting modified parameters.
    This function is designed to be used as an `on_retry` callback for the
    `@retry` decorator.
    """
    log_content = getattr(exception, "stdout", "") + getattr(exception, "stderr", "")
    job = kwargs.get("job")
    if not isinstance(job, DFTJob):
        return None

    current_params = job.params.model_copy()
    updated_fields = {}

    if "convergence NOT achieved" in log_content and current_params.mixing_beta > 0.1:
        updated_fields["mixing_beta"] = current_params.mixing_beta / 2
        logger.info(f"Convergence failed. Halving mixing_beta to {updated_fields['mixing_beta']}.")

    if "Cholesky" in log_content and current_params.diagonalization == "david":
        updated_fields["diagonalization"] = "cg"
        logger.info("Cholesky issue detected. Switching diagonalization to 'cg'.")

    if updated_fields:
        new_params = current_params.model_copy(update=updated_fields)
        job.params = new_params
        return {"job": job}

    return None


class DFTRunner:
    """
    Executes and manages a `DFTJob`.

    Args:
        input_generator: An instance of `QEInputGenerator` to use for
                         creating input files.
        process_runner: An instance of `QEProcessRunner` to use for
                        running the DFT calculation.
        output_parser: An instance of `QEOutputParser` to use for
                       parsing the output of the calculation.
    """

    def __init__(
        self,
        input_generator: QEInputGenerator,
        process_runner: QEProcessRunner,
        output_parser: QEOutputParser,
    ) -> None:
        self.input_generator = input_generator
        self.process_runner = process_runner
        self.output_parser = output_parser

    @retry(
        attempts=3,
        delay=5.0,
        exceptions=(DFTCalculationError, subprocess.CalledProcessError),
        on_retry=dft_retry_handler,
    )
    def run(self, job: DFTJob) -> DFTResult:
        """
        Runs a DFTJob, handling transient errors with the @retry decorator.
        """
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                work_dir = Path(temp_dir)
                input_path = work_dir / "espresso.pwi"
                output_path = work_dir / "espresso.pwo"
                self.input_generator.prepare_input_files(work_dir, job.atoms, job.params)
                self.process_runner.execute(input_path, output_path)
                result = self.output_parser.parse(output_path, job.job_id)
                logger.info(f"DFT job {job.job_id} succeeded.")
                return result
        except (subprocess.CalledProcessError, DFTCalculationError) as e:
            logger.exception(
                f"DFT job {job.job_id} failed.",
                extra={
                    "job_id": job.job_id,
                    "stdout": getattr(e, "stdout", ""),
                    "stderr": getattr(e, "stderr", ""),
                },
            )
            raise
