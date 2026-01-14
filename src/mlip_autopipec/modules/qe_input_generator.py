from pathlib import Path

from ase.calculators.espresso import Espresso, EspressoProfile

from mlip_autopipec.schemas.dft import DFTInput
from mlip_autopipec.settings import settings


class QEInputGenerator:
    """
    Generates Quantum Espresso input files.

    This class takes a DFTInput object and writes a Quantum Espresso
    input file to a specified directory.
    """

    def __init__(self, dft_input: DFTInput) -> None:
        """
        Initializes the QEInputGenerator.

        Args:
            dft_input: The input for the DFT calculation.
        """
        self.dft_input = dft_input

    def write_input(self, directory: Path) -> None:
        """
        Writes the QE input file to the given directory.

        Args:
            directory: The directory to write the input file to.
        """
        profile = EspressoProfile(command=settings.qe_command, pseudo_dir=directory / "pseudos")
        input_data = self.dft_input.dft_params.model_dump(
            exclude={"pseudopotentials", "mixing_beta"}
        )
        if self.dft_input.dft_params.mixing_beta is not None:
            input_data.setdefault("ELECTRONS", {})
            input_data["ELECTRONS"]["mixing_beta"] = self.dft_input.dft_params.mixing_beta
        calc = Espresso(
            profile=profile,
            input_data=input_data,
            pseudopotentials=self.dft_input.dft_params.pseudopotentials,
            directory=directory,
        )
        calc.write_inputfiles(self.dft_input.atoms, {})
