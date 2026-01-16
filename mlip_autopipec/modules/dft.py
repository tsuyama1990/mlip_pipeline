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
from mlip_autopipec.utils.resilience import QERetryHandler

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
SSSP_DATA_PATH = DATA_DIR / "sssp_p_data.json"
MAGNETIC_ELEMENTS = {"Fe", "Co", "Ni", "Cr", "Mn"}


class QEInputGenerator:
    """
    Creates Quantum Espresso input files from a `DFTInputParameters` model.

    This class translates a validated Pydantic model of DFT parameters into the
    specific `pw.x` input file format required by Quantum Espresso, using the
    ASE `Espresso` calculator as a backend.
    """

    def __init__(
        self, profile: EspressoProfile, pseudopotentials_path: Path | None
    ) -> None:
        """
        Initializes the QEInputGenerator.

        Args:
            profile: An ASE `EspressoProfile` configured with the path to the
                     `pw.x` executable.
            pseudopotentials_path: The path to the directory containing the
                                   pseudopotential files.
        """
        self.profile = profile
        self.pseudopotentials_path = pseudopotentials_path

    def prepare_input_files(
        self, work_dir: Path, atoms: Atoms, params: DFTInputParameters
    ) -> None:
        """
        Writes the `espresso.pwi` input file to the working directory.

        Args:
            work_dir: The directory where the input file will be written.
            atoms: The `ase.Atoms` object for the calculation.
            params: The `DFTInputParameters` for the calculation.
        """
        input_data = self._build_input_data(work_dir, params)
        calculator = Espresso(
            profile=self.profile,
            directory=str(work_dir),
            kpts=params.k_points,
            pseudopotentials=params.pseudopotentials.model_dump(),
            input_data=input_data,
        )
        calculator.write_inputfiles(
            atoms, properties=["energy", "forces", "stress"]
        )

    def _build_input_data(
        self, work_dir: Path, params: DFTInputParameters
    ) -> dict:
        """Constructs the nested dictionary for ASE's `Espresso` calculator."""
        pseudo_dir = (
            str(self.pseudopotentials_path)
            if self.pseudopotentials_path
            else "."
        )
        input_data = {
            "control": {
                "calculation": params.calculation_type,
                "pseudo_dir": pseudo_dir,
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
            input_data["system"].update(
                {
                    "occupations": "smearing",
                    "smearing": params.smearing.smearing_type,
                    "degauss": params.smearing.degauss,
                }
            )
        if params.magnetism:
            input_data["system"]["nspin"] = params.magnetism.nspin
            for el, mom in params.magnetism.starting_magnetization.items():
                input_data["system"][f"starting_magnetization({el})"] = mom
        return input_data


class QEProcessRunner:
    """
    Executes a Quantum Espresso calculation in a secure subprocess.

    This class is responsible for running the `pw.x` executable. It constructs
    the command using ASE's `EspressoProfile` and executes it in a sandboxed
    environment, capturing stdout and stderr.
    """

    def __init__(self, profile: EspressoProfile) -> None:
        """
        Initializes the QEProcessRunner.

        Args:
            profile: An ASE `EspressoProfile` configured with the path to the
                     `pw.x` executable.
        """
        self.profile = profile

    def execute(self, input_path: Path, output_path: Path) -> None:
        """
        Runs `pw.x` using the provided input file.

        Args:
            input_path: Path to the `espresso.pwi` input file.
            output_path: Path where the stdout of the `pw.x` run will be
                         written.

        Raises:
            FileNotFoundError: If the `pw.x` executable is not found.
            subprocess.CalledProcessError: If `pw.x` returns a non-zero exit
                                           code.
        """
        command = self.profile.get_command(inputfile=str(input_path))
        try:
            with input_path.open() as stdin_f, output_path.open("w") as stdout_f:
                # SECURITY: The `command` is generated by ASE's EspressoProfile
                # and is not derived from user input, mitigating command
                # injection risks. `shell=False` is explicitly set as a best
                # practice.
                process = subprocess.run(
                    command,
                    shell=False,
                    check=True,
                    capture_output=True,
                    text=True,
                    stdin=stdin_f,
                )
                stdout_f.write(process.stdout)
        except FileNotFoundError as e:
            logger.error(f"QE executable not found. Details: {e}")
            raise
        except subprocess.CalledProcessError as e:
            logger.error(
                "QE subprocess failed.\\n"
                f"  Exit Code: {e.returncode}\\n"
                f"  Stdout: {e.stdout}\\n"
                f"  Stderr: {e.stderr}"
            )
            raise


class QEOutputParser:
    """
    Parses a Quantum Espresso output file into a `DFTResult` object.

    This class uses the ASE `read` function with the `espresso-out` format to
    extract the final energy, forces, and stress from a `pw.x` output file.
    It then wraps this data in a validated `DFTResult` Pydantic model.
    """

    def __init__(self, reader: Any = ase_read) -> None:
        """
        Initializes the QEOutputParser.

        Args:
            reader: A callable (like `ase.io.read`) that can parse QE output
                    files. This is dependency-injected for testability.
        """
        self.reader = reader

    def parse(self, output_path: Path, job_id: Any) -> DFTResult:
        """
        Parses the `espresso.pwo` output file of a successful QE run.

        Args:
            output_path: The path to the QE output file.
            job_id: The unique identifier for the DFT job.

        Returns:
            A `DFTResult` object containing the parsed energy, forces, and
            stress.

        Raises:
            DFTCalculationError: If the output file cannot be parsed.
        """
        try:
            result_atoms = self.reader(output_path, format="espresso-out")
            energy = result_atoms.get_potential_energy()
            forces = result_atoms.get_forces()
            stress = result_atoms.get_stress()

            return DFTResult(
                job_id=job_id,
                energy=energy,
                forces=forces,
                stress=stress,
            )
        except (OSError, IndexError) as e:
            msg = f"Failed to parse QE output file: {output_path}"
            raise DFTCalculationError(msg) from e


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
        """
        return {el: self._sssp_data[el]["filename"] for el in elements if el in self._sssp_data}


class DFTRunner:
    """
    Executes and manages a `DFTJob`.

    This class is responsible for the practical execution of a DFT calculation.
    It takes a `DFTJob` object, which contains the atomic structure and all
    necessary parameters, and orchestrates the process of running the external
    DFT code. This includes generating input files, running the calculation in a
    subprocess, parsing the output, and handling retries for convergence
    errors. It is designed to be a stateful, single-job runner.
    """

    def __init__(
        self,
        input_generator: QEInputGenerator,
        process_runner: QEProcessRunner,
        output_parser: QEOutputParser,
        retry_handler: QERetryHandler,
        max_retries: int = 3,
    ) -> None:
        self.input_generator = input_generator
        self.process_runner = process_runner
        self.output_parser = output_parser
        self.retry_handler = retry_handler
        self.max_retries = max_retries

    def run(self, job: DFTJob) -> DFTResult:
        """Runs a DFTJob and returns a DFTResult."""
        for attempt in range(self.max_retries):
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    work_dir = Path(temp_dir)
                    input_path = work_dir / "espresso.pwi"
                    output_path = work_dir / "espresso.pwo"
                    self.input_generator.prepare_input_files(work_dir, job.atoms, job.params)
                    self.process_runner.execute(input_path, output_path)
                    result = self.output_parser.parse(output_path, job.job_id)
                    logger.info(f"DFT job {job.job_id} succeeded on attempt {attempt + 1}.")
                    return result
            except (subprocess.CalledProcessError, DFTCalculationError) as e:
                logger.exception(
                    f"DFT job {job.job_id} failed on attempt {attempt + 1}.",
                    extra={"job_id": job.job_id, "attempt": attempt + 1},
                )
                log_content = e.stdout + "\\n" + e.stderr if hasattr(e, "stdout") else ""
                new_params = self.retry_handler.handle_convergence_error(
                    log_content, job.params.model_dump()
                )
                if new_params and attempt < self.max_retries - 1:
                    updated_params = job.params.model_dump()
                    updated_params.update(new_params)
                    job.params = DFTInputParameters.model_validate(updated_params)
                    logger.info("Retrying with modified parameters...")
                else:
                    raise DFTCalculationError(
                        f"DFT calculation failed for job {job.job_id} after all retries.",
                        stdout=e.stdout,
                        stderr=e.stderr,
                    ) from e
        msg = f"DFT job {job.job_id} failed unexpectedly."
        raise DFTCalculationError(msg)
