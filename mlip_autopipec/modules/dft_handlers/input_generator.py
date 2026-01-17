"""
This module contains the `QEInputGenerator` class.
"""

from pathlib import Path

from ase.atoms import Atoms
from ase.calculators.espresso import Espresso, EspressoProfile

from mlip_autopipec.config.models import DFTInputParameters


class QEInputGenerator:
    """
    Creates Quantum Espresso input files from a `DFTInputParameters` model.

    This class translates a validated Pydantic model of DFT parameters into the
    specific `pw.x` input file format required by Quantum Espresso, using the
    ASE `Espresso` calculator as a backend.
    """

    def __init__(self, profile: EspressoProfile, pseudopotentials_path: Path | None) -> None:
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

    def prepare_input_files(self, work_dir: Path, atoms: Atoms, params: DFTInputParameters) -> None:
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
        self._validate_pseudopotentials(params)
        calculator.write_inputfiles(atoms, properties=["energy", "forces", "stress"])

    def _validate_pseudopotentials(self, params: DFTInputParameters) -> None:
        """Checks if all required pseudopotential files exist."""
        if not self.pseudopotentials_path:
            raise FileNotFoundError("Pseudopotential directory not specified.")

        for pseudo_filename in params.pseudopotentials.root.values():
            pseudo_path = self.pseudopotentials_path / pseudo_filename
            if not pseudo_path.is_file():
                raise FileNotFoundError(
                    f"Pseudopotential file not found: {pseudo_path}"
                )

    def _build_input_data(self, work_dir: Path, params: DFTInputParameters) -> dict:
        """Constructs the nested dictionary for ASE's `Espresso` calculator."""
        pseudo_dir = str(self.pseudopotentials_path) if self.pseudopotentials_path else "."
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
