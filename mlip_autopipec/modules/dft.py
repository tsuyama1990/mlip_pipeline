# FIXME: The above comment is a temporary workaround for a ruff bug.
# It should be removed once the bug is fixed.
# For more information, see: https://github.com/astral-sh/ruff/issues/10515
"""
This module contains the `DFTFactory`, the core component for running and
managing DFT calculations in the MLIP-AutoPipe workflow.

It is designed as a robust, fault-tolerant service that abstracts the
complexities of DFT calculations, providing a simple interface to the rest of
the application.
"""

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ase.atoms import Atoms
from ase.calculators.espresso import Espresso, EspressoProfile
from ase.io import read as ase_read

from mlip_autopipec.config.models import (
    CutoffConfig,
    DFTInputParameters,
    DFTJob,
    DFTResult,
    MagnetismConfig,
    SmearingConfig,
)
from mlip_autopipec.exceptions import DFTCalculationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# --- Constants ---
# Path to the SSSP data file, assumed to be in the `data` directory
DATA_DIR = Path(__file__).parent.parent / "data"
SSSP_DATA_PATH = DATA_DIR / "sssp_p_data.json"

# Set of elements that are typically magnetic
MAGNETIC_ELEMENTS = {"Fe", "Co", "Ni", "Cr", "Mn"}


class DFTFactory:
    """
    A factory for creating and executing DFT calculations using Quantum Espresso.

    This class encapsulates the logic for:
    - Automatically determining calculation parameters using heuristics.
    - Generating input files for Quantum Espresso.
    - Executing the DFT calculation as a subprocess.
    - Parsing the output to extract energy, forces, and stress.
    - Implementing a retry mechanism with auto-recovery for common failures.
    """

    def __init__(
        self,
        qe_executable_path: str,
        max_retries: int = 3,
        pseudopotentials_path: Path | None = None,
        pseudo_dir: Path | None = None,
    ):
        """
        Initializes the DFTFactory.

        Args:
            qe_executable_path: The full path to the Quantum Espresso `pw.x`
                                executable.
            max_retries: The maximum number of times to retry a failed
                         calculation.
            pseudopotentials_path: The path to the directory containing the
                                   pseudopotential files. If `None`, it is
                                   assumed they are in the working directory.
            pseudo_dir: The path to the pseudopotentials directory for
                        EspressoProfile.
        """
        self.profile = EspressoProfile(
            command=qe_executable_path,
            pseudo_dir=pseudo_dir,
        )
        self.pseudopotentials_path = pseudopotentials_path
        self.max_retries = max_retries
        self._sssp_data = self._load_sssp_data()

    def _load_sssp_data(self) -> dict[str, Any]:
        """Loads the SSSP pseudopotential data from the JSON file."""
        try:
            with open(SSSP_DATA_PATH) as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"SSSP data file not found at: {SSSP_DATA_PATH}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Error decoding SSSP data file: {SSSP_DATA_PATH}")
            raise

    def run(self, atoms: Atoms) -> DFTResult:
        """
        Orchestrates the DFT calculation for a given atomic structure.

        This method manages the entire lifecycle of a DFT job, including
        parameter generation, execution, parsing, and resilience.

        Args:
            atoms: An `ase.Atoms` object representing the structure to be
                   calculated.

        Returns:
            A `DFTResult` object containing the converged energy, forces, and
            stress.

        Raises:
            DFTCalculationError: If the calculation fails after all retry
                                 attempts.
        """
        params = self._get_heuristic_parameters(atoms)
        job = DFTJob(atoms=atoms, params=params)

        for attempt in range(self.max_retries):
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    work_dir = Path(temp_dir)
                    input_path = work_dir / "espresso.pwi"
                    output_path = work_dir / "espresso.pwo"

                    # Prepare and execute the calculation
                    self._prepare_input_files(
                        work_dir,
                        job.atoms,
                        job.params,
                    )
                    self._execute_dft(input_path, output_path)

                    # If successful, parse and return the result
                    result = self._parse_output(output_path, job.job_id)
                    logger.info(
                        f"DFT calculation for job {job.job_id} succeeded on attempt {attempt + 1}."
                    )
                    return result

            except subprocess.CalledProcessError as e:
                logger.warning(
                    f"DFT calculation for job {job.job_id} failed on attempt {attempt + 1}."
                )
                log_content = e.stdout + "\n" + e.stderr
                new_params_dict = self._handle_convergence_error(
                    log_content,
                    job.params.model_dump(),
                )

                if new_params_dict and attempt < self.max_retries - 1:
                    job.params = DFTInputParameters(**new_params_dict)
                    logger.info("Retrying with modified parameters...")
                else:
                    raise DFTCalculationError(
                        f"DFT calculation failed for job {job.job_id} after all retries.",
                        stdout=e.stdout,
                        stderr=e.stderr,
                    ) from e
        # This part should be unreachable, but it's here for safety.
        raise DFTCalculationError(f"DFT calculation for job {job.job_id} failed unexpectedly.")

    def _prepare_input_files(
        self,
        work_dir: Path,
        atoms: Atoms,
        params: DFTInputParameters,
    ) -> None:
        """
        Generates the Quantum Espresso input file.

        Args:
            work_dir: The directory where the input file will be written.
            atoms: The `ase.Atoms` object.
            params: The validated `DFTInputParameters` for the calculation.
        """
        input_data = {
            "control": {
                "calculation": params.calculation_type,
                "pseudo_dir": str(self.pseudopotentials_path)
                if self.pseudopotentials_path
                else ".",
                "outdir": str(work_dir),
            },
            "system": {
                "ecutwfc": params.cutoffs.wavefunction,
                "ecutrho": params.cutoffs.density,
            },
            "electrons": {
                "mixing_beta": params.mixing_beta,
                "diagonalization": params.diagonalization,
            },
        }

        if params.smearing:
            input_data["system"]["occupations"] = "smearing"
            input_data["system"]["smearing"] = params.smearing.smearing_type
            input_data["system"]["degauss"] = params.smearing.degauss

        if params.magnetism:
            input_data["system"]["nspin"] = params.magnetism.nspin
            for element, moment in params.magnetism.starting_magnetization.items():
                input_data["system"][f"starting_magnetization({element})"] = moment

        calculator = Espresso(
            profile=self.profile,
            directory=str(work_dir),
            kpts=params.k_points,
            pseudopotentials=params.pseudopotentials,
            input_data=input_data,
        )
        calculator.write_inputfiles(atoms, properties=["energy", "forces", "stress"])

    def _execute_dft(self, input_path: Path, output_path: Path) -> None:
        """
        Executes the Quantum Espresso calculation as a subprocess.

        Args:
            input_path: Path to the QE input file.
            output_path: Path where the QE output file will be written.

        Raises:
            subprocess.CalledProcessError: If the `pw.x` process returns a
                                           non-zero exit code.
        """
        command = self.profile.get_command(inputfile=str(input_path))
        with (
            open(input_path) as stdin_file,
            open(
                output_path,
                "w",
            ) as stdout_file,
        ):
            process = subprocess.run(
                command,
                shell=False,
                check=True,
                capture_output=True,  # Capture stderr
                text=True,
                stdin=stdin_file,
            )
            stdout_file.write(process.stdout)
            if process.returncode != 0:
                raise subprocess.CalledProcessError(
                    process.returncode,
                    cmd=command,
                    output=process.stdout,
                    stderr=process.stderr,
                )

    def _parse_output(self, output_path: Path, job_id: Any) -> DFTResult:
        """
        Parses the output file of a successful QE run.

        Args:
            output_path: Path to the QE output file.
            job_id: The unique identifier for the job.

        Returns:
            A `DFTResult` object populated with the extracted data.
        """
        try:
            result_atoms = ase_read(output_path, format="espresso-out")
            energy = result_atoms.info["energy"]
            forces = result_atoms.info["forces"]
            stress = result_atoms.info["stress"]

            return DFTResult(
                job_id=job_id,
                energy=energy,
                forces=forces,
                stress=stress,
            )
        except Exception as e:
            raise DFTCalculationError(f"Failed to parse QE output file: {output_path}") from e

    def _handle_convergence_error(
        self,
        log_content: str,
        current_params: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Diagnoses a convergence error and suggests modified parameters.

        Args:
            log_content: The stdout/stderr from the failed QE run.
            current_params: The dictionary of parameters used in the failed run.

        Returns:
            A new dictionary of parameters with modified values for the next
            retry attempt, or `None` if no recovery strategy is found.
        """
        new_params = current_params.copy()
        if "convergence NOT achieved" in log_content:
            current_beta = new_params.get("mixing_beta", 0.7)
            new_beta = round(current_beta * 0.5, 2)  # Reduce by 50%
            if new_beta > 0.01:
                new_params["mixing_beta"] = new_beta
                logger.info(f"Convergence failed. Reducing mixing_beta to {new_beta}")
                return new_params

        if "Cholesky" in log_content:
            if new_params.get("diagonalization") != "cg":
                new_params["diagonalization"] = "cg"
                logger.info("Cholesky error detected. Switching to 'cg' diagonalization.")
                return new_params

        return None  # No recovery strategy found

    def _get_heuristic_parameters(self, atoms: Atoms) -> DFTInputParameters:
        """
        Determines a set of reasonable DFT parameters based on the input
        structure.

        Args:
            atoms: The `ase.Atoms` object.

        Returns:
            A validated `DFTInputParameters` object.
        """
        elements = set(atoms.get_chemical_symbols())
        cutoffs = self._get_heuristic_cutoffs(elements)
        k_points = self._get_heuristic_k_points(atoms)
        smearing = SmearingConfig()  # Always use smearing for robustness
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
                raise ValueError(f"No SSSP data found for element: {element}")
        return CutoffConfig(wavefunction=max_wfc, density=max_rho)

    def _get_heuristic_k_points(self, atoms: Atoms) -> tuple[int, int, int]:
        """
        Calculates a k-point grid based on a target density.
        """
        k_density = 6.0  # A reasonable default for many materials
        lengths = atoms.cell.lengths()
        k_points = []
        for length in lengths:
            if length > 0:
                k_points.append(max(1, int(k_density / length) + 1))
            else:
                k_points.append(1)  # For non-periodic directions
        return tuple(k_points)  # type: ignore

    def _get_heuristic_magnetism(
        self,
        elements: set,
    ) -> MagnetismConfig | None:
        """
        Enables magnetism if magnetic elements are present.
        """
        if any(el in MAGNETIC_ELEMENTS for el in elements):
            return MagnetismConfig(starting_magnetization=dict.fromkeys(elements, 0.5))
        return None

    def _get_pseudopotentials(self, elements: set) -> dict[str, str]:
        """
        Retrieves the pseudopotential filenames for the given elements.
        """
        return {el: self._sssp_data[el]["filename"] for el in elements if el in self._sssp_data}
