"""
This module contains the components for creating, managing, and running DFT
calculations, forming a key part of the data generation pipeline.
"""

import logging
import subprocess
import tempfile
from pathlib import Path

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
from mlip_autopipec.exceptions import DFTCalculationException
from mlip_autopipec.utils.data_loader import load_sssp_data
from mlip_autopipec.utils.resilience import retry

from .dft_handlers.input_generator import QEInputGenerator
from .dft_handlers.output_parser import QEOutputParser
from .dft_handlers.process_runner import QEProcessRunner
from .dft_handlers.retry import dft_retry_handler

logger = logging.getLogger(__name__)

# Constants
MAGNETIC_ELEMENTS = {"Fe", "Co", "Ni", "Cr", "Mn"}


class DFTHeuristics:
    """
    A dedicated class for generating sensible DFT parameters based on heuristics.

    This class encapsulates the domain-specific knowledge required to choose
    parameters like k-points, cutoffs, and magnetism settings for a given
    atomic structure. It is designed to be a stateless utility class.

    Args:
        sssp_data_path: The path to the JSON file containing the SSSP data.
    """

    def __init__(self, sssp_data_path: Path) -> None:
        self._sssp_data = load_sssp_data(sssp_data_path)

    def get_heuristic_parameters(self, atoms: Atoms) -> DFTInputParameters:
        """
        Determines a set of reasonable DFT parameters for a given structure.

        Args:
            atoms: The atomic structure.

        Returns:
            DFTInputParameters: The generated parameters.
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

    Args:
        heuristics: An instance of `DFTHeuristics` to use for parameter generation.
    """

    def __init__(self, heuristics: DFTHeuristics) -> None:
        self.heuristics = heuristics

    def create_job(self, atoms: Atoms) -> DFTJob:
        """
        Creates a DFTJob with heuristic parameters for a given atomic structure.
        """
        params = self.heuristics.get_heuristic_parameters(atoms)
        return DFTJob(atoms=atoms, params=params)


class DFTRunner:
    """
    Executes and manages a `DFTJob`.

    Args:
        input_generator: An instance of `QEInputGenerator`.
        process_runner: An instance of `QEProcessRunner`.
        output_parser: An instance of `QEOutputParser`.
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
        exceptions=(DFTCalculationException, subprocess.CalledProcessError),
        on_retry=dft_retry_handler,
    )
    def run(self, job: DFTJob) -> DFTResult:
        """
        Runs a DFTJob, handling transient errors with the @retry decorator.

        Args:
            job: The DFT job configuration.

        Returns:
            DFTResult: The parsed results of the calculation.

        Raises:
            DFTCalculationException: If the calculation fails after retries.
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

        except DFTCalculationException:
            # Let domain-specific errors propagate directly
            raise

        except subprocess.CalledProcessError as e:
            # Wrap low-level subprocess errors with context
            raise DFTCalculationException(
                f"DFT subprocess failed for job {job.job_id}",
                stdout=getattr(e, "stdout", ""),
                stderr=getattr(e, "stderr", ""),
            ) from e

        except Exception as e:
            # Catch-all for unexpected runtime errors to prevent crash
            logger.exception(f"Unexpected error executing DFT job {job.job_id}")
            raise DFTCalculationException(f"Unexpected error: {e}") from e
