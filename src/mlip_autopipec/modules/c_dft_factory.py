import subprocess
import tempfile
from pathlib import Path

from ase.calculators.espresso import Espresso, EspressoProfile
from mlip_autopipec.schemas.dft import DFTInput, DFTOutput
from mlip_autopipec.settings import settings


class QERunner:
    """A wrapper for Quantum Espresso to run DFT calculations."""

    def __init__(self, max_retries: int = 3, keep_temp_dir: bool = False):
        self.max_retries = max_retries
        self.keep_temp_dir = keep_temp_dir

    def run(self, dft_input: DFTInput) -> DFTOutput:
        """Runs a DFT calculation and returns the output."""
        for attempt in range(self.max_retries):
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                if self.keep_temp_dir:
                    tmpdir_path = Path(tempfile.mkdtemp())
                pseudo_dir = tmpdir_path / "pseudos"
                pseudo_dir.mkdir()
                profile = EspressoProfile(
                    command=settings.qe_command, pseudo_dir=pseudo_dir
                )  # type: ignore[no-untyped-call]
                input_data = dft_input.dft_params.model_dump(
                    exclude={"pseudopotentials", "mixing_beta"}
                )
                input_data.setdefault("ELECTRONS", {})
                if dft_input.dft_params.mixing_beta is not None:
                    input_data["ELECTRONS"]["mixing_beta"] = (
                        dft_input.dft_params.mixing_beta
                    )
                calc = Espresso(  # type: ignore[no-untyped-call]
                    profile=profile,
                    input_data=input_data,
                    pseudopotentials=dft_input.dft_params.pseudopotentials,
                    directory=tmpdir_path,
                )
                calc.write_inputfiles(dft_input.atoms, {})  # type: ignore[no-untyped-call]

                command = [
                    settings.qe_command,
                    "-in",
                    str(tmpdir_path / "espresso.pwi"),
                ]
                process = subprocess.run(  # noqa: S603
                    command, capture_output=True, text=True, check=False
                )

                if process.returncode == 0 and "!" in process.stdout:
                    return self._parse_output(process.stdout)

                dft_input = self._recover(dft_input)

        error_message = "Quantum Espresso failed after multiple retries."
        raise RuntimeError(error_message)

    def _recover(self, dft_input: DFTInput) -> DFTInput:
        """Modifies DFT parameters to attempt recovery from a failed run."""
        new_params = dft_input.dft_params.model_copy(deep=True)
        new_params.mixing_beta = 0.3
        return DFTInput(atoms=dft_input.atoms, dft_params=new_params)

    def _parse_output(self, qe_output: str) -> DFTOutput:
        """Parses the stdout of a Quantum Espresso calculation."""
        total_energy = 0.0
        forces = []
        stress = []
        parsing_forces = False
        parsing_stress = False
        stress_lines = []

        for line in qe_output.splitlines():
            if "!    total energy" in line:
                total_energy = float(line.split()[-2])
            elif "Forces acting on atoms" in line:
                parsing_forces = True
            elif parsing_forces and "atom" in line:
                parts = line.split()
                forces.append([float(parts[6]), float(parts[7]), float(parts[8])])
            elif "total stress" in line:
                parsing_forces = False
                parsing_stress = True
                stress_lines = []
            elif parsing_stress and len(stress_lines) < 3:
                stress_lines.append(line)

        if stress_lines:
            stress = np.array(
                [list(map(float, line.split())) for line in stress_lines]
            )
            stress = (stress * 1e-1).tolist()  # Convert from kbar to GPa

        return DFTOutput(total_energy=total_energy, forces=forces, stress=stress)
