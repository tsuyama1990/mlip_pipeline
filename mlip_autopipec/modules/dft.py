"""
This module contains the `DFTFactory` and its dependencies, which form the
core component for running and managing DFT calculations.
"""

import json
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
    SmearingConfig,
)
from typing import Any, Dict

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


class DFTJobFactory:
    """
    Creates `DFTJob` objects with heuristic-driven parameters.

    This class is responsible for taking a raw `ase.Atoms` object and
    determining a sensible set of DFT parameters for it based on a set of
    pre-defined heuristics (e.g., k-point density, cutoffs from SSSP).
    Its sole purpose is to create a validated `DFTJob` that can be
    executed by a `DFTRunner`.
    """

    def __init__(self) -> None:
        self._sssp_data = self._load_sssp_data()

    def create_job(self, atoms: Atoms) -> DFTJob:
        """
        Creates a DFTJob with heuristic parameters for a given atomic structure.

        Args:
            atoms: An `ase.Atoms` object representing the structure to be
                   calculated.

        Returns:
            A `DFTJob` object containing the atoms and the generated DFT
            parameters, ready for execution.
        """
        params = self._get_heuristic_parameters(atoms)
        return DFTJob(atoms=atoms, params=params)

    def _load_sssp_data(self) -> dict[str, Any]:
        """Loads the SSSP pseudopotential data from the JSON file."""
        try:
            with SSSP_DATA_PATH.open() as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"SSSP data file not found at: {SSSP_DATA_PATH}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Error decoding SSSP data file: {SSSP_DATA_PATH}")
            raise

    def _get_heuristic_parameters(self, atoms: Atoms) -> DFTInputParameters:
        """
        Determines a set of reasonable DFT parameters based on the input
        structure.
        """
        elements = set(atoms.get_chemical_symbols())
        cutoffs = self._get_heuristic_cutoffs(elements)
        k_points = self._get_heuristic_k_points(atoms)
        smearing = SmearingConfig()
        magnetism = self._get_heuristic_magnetism(elements)
        pseudos = self._get_pseudopotentials(elements)

        return DFTInputParameters(
            pseudopotentials=pseudos,
            cutoffs=cutoffs,
            k_points=k_points,
            smearing=smearing,
            magnetism=magnetism,
        )

    def _get_heuristic_cutoffs(self, elements: set) -> CutoffConfig:
        """
        Determines the wavefunction and density cutoffs from the SSSP data.
        """
        max_wfc = 0.0
        max_rho = 0.0
        for element in elements:
            if element in self._sssp_data:
                max_wfc = max(max_wfc, self._sssp_data[element]["cutoff_wfc"])
                max_rho = max(max_rho, self._sssp_data[element]["cutoff_rho"])
            else:
                msg = f"No SSSP data found for element: {element}"
                raise ValueError(msg)
        return CutoffConfig(wavefunction=max_wfc, density=max_rho)

    def _get_heuristic_k_points(self, atoms: Atoms) -> tuple[int, int, int]:
        """
        Calculates a k-point grid based on a target density.
        """
        k_density = 6.0
        lengths = atoms.cell.lengths()
        k_points = []
        for length in lengths:
            if length > 0:
                k_points.append(max(1, int(k_density / length) + 1))
            else:
                k_points.append(1)
        return tuple(k_points)

    def _get_heuristic_magnetism(self, elements: set) -> MagnetismConfig | None:
        """
        Enables magnetism if magnetic elements are present.
        """
        if any(el in MAGNETIC_ELEMENTS for el in elements):
            return MagnetismConfig(starting_magnetization=dict.fromkeys(elements, 0.5))
        return None

    def _get_pseudopotentials(self, elements: set) -> dict[str, str]:
        """
        Retrieves the pseudopotential filenames for the given elements.

        Args:
            elements: A set of chemical symbols.

        Returns:
            A dictionary mapping chemical symbols to their corresponding
            pseudopotential filenames.
        """
        return {el: self._sssp_data[el]["filename"] for el in elements if el in self._sssp_data}


def dft_retry_handler(exception: Exception, kwargs: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    Handles specific DFT convergence errors and suggests modified parameters.
    """
    log_content = ""
    if hasattr(exception, "stdout"):
        log_content += exception.stdout
    if hasattr(exception, "stderr"):
        log_content += exception.stderr

    job = kwargs.get("job")
    if not job or not isinstance(job, DFTJob):
        return None

    current_params = job.params
    new_params = {}

    if "convergence NOT achieved" in log_content:
        if current_params.mixing_beta > 0.1:
            new_params["mixing_beta"] = current_params.mixing_beta / 2
            logger.info(f"Convergence failed. Halving mixing_beta to {new_params['mixing_beta']}.")

    if "Cholesky" in log_content:
        if current_params.diagonalization == "david":
            new_params["diagonalization"] = "cg"
            logger.info("Cholesky issue detected. Switching diagonalization to 'cg'.")

    if new_params:
        updated_params = current_params.model_dump()
        updated_params.update(new_params)
        job.params = DFTInputParameters.model_validate(updated_params)
        return {"job": job}

    return None


class DFTRunner:
    """
    Executes and manages a `DFTJob`.
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
        Runs a DFTJob and returns a DFTResult.
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
            logger.error(
                f"DFT job {job.job_id} failed.",
                extra={
                    "job_id": job.job_id,
                    "stdout": e.stdout if hasattr(e, "stdout") else "",
                    "stderr": e.stderr if hasattr(e, "stderr") else "",
                },
            )
            raise
