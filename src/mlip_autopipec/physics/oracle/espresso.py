import logging
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from mlip_autopipec.config import DFTConfig
from mlip_autopipec.utils.parsers import DFTConvergenceError, QEParser

logger = logging.getLogger(__name__)


class EspressoRunner:
    def __init__(self, config: DFTConfig) -> None:
        self.config = config

    def run_single(self, atoms: Atoms) -> Atoms:
        params = self._get_base_params()

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            for attempt in range(self.config.max_retries + 1):
                try:
                    return self._execute_calculation(atoms, params, tmp_path)
                except DFTConvergenceError:
                    if attempt == self.config.max_retries:
                        logger.exception("Max retries reached for DFT calculation.")
                        raise
                    logger.warning(f"SCF failed. Retrying (Attempt {attempt + 1})...")
                    params = self._adjust_params_for_healing(params)

            msg = "Should not reach here"
            raise RuntimeError(msg)

    def _get_base_params(self) -> dict[str, Any]:
        return {
            "ecutwfc": self.config.ecutwfc,
            "kspacing": self.config.kspacing,
            "mixing_beta": 0.7,
            "diagonalization": "david",
            "smearing": "fermi-dirac",
            "sigma": 0.02,
        }

    def _execute_calculation(self, atoms: Atoms, params: dict[str, Any], work_dir: Path) -> Atoms:
        input_file = work_dir / "pw.in"
        output_file = work_dir / "pw.out"

        self._generate_input_file(atoms, params, input_file)

        # Using shell=True for simple redirection as per spec
        full_cmd = f"{self.config.command} < {input_file} > {output_file}"

        logger.info(f"Running DFT in {work_dir}: {full_cmd}")

        # We don't check return code automatically because QE might exit 0 even on failure,
        # or we want to parse the output for specific errors.
        subprocess.run(  # noqa: S602
            full_cmd,
            shell=True,
            cwd=work_dir,
            capture_output=False,
            check=False,
        )

        if not output_file.exists():
            msg = "DFT output file not generated"
            raise RuntimeError(msg)

        content = output_file.read_text()
        parser = QEParser(content)
        parser.check_convergence()

        # If converged, parse results
        energy = parser.parse_energy()
        forces = parser.parse_forces()
        stress = parser.parse_stress()

        atoms_out = cast(Atoms, atoms.copy())  # type: ignore[no-untyped-call]
        calc = SinglePointCalculator(  # type: ignore[no-untyped-call]
            atoms_out,
            energy=energy,
            forces=forces,
            stress=stress,
        )
        atoms_out.calc = calc
        return atoms_out

    def _generate_input_file(self, atoms: Atoms, params: dict[str, Any], filepath: Path) -> None:
        input_data = {
            "control": {
                "calculation": "scf",
                "restart_mode": "from_scratch",
                "pseudo_dir": ".",  # Assuming pseudos are copied or handled by environment
                "tprnfor": True,
                "tstress": True,
                "disk_io": "none",  # Minimize I/O
            },
            "system": {
                "ecutwfc": params["ecutwfc"],
                "occupations": "smearing",
                "smearing": params["smearing"],
                "degauss": params["sigma"],
            },
            "electrons": {
                "diagonalization": params["diagonalization"],
                "mixing_beta": params["mixing_beta"],
            },
        }

        # Need to handle pseudopotentials path logic if needed,
        # but sticking to basic implementation for now.

        write(
            filepath,
            atoms,
            format="espresso-in",
            input_data=input_data,
            pseudopotentials=self.config.pseudopotentials,
            kspacing=params["kspacing"],
        )

    def _adjust_params_for_healing(self, params: dict[str, Any]) -> dict[str, Any]:
        new_params = params.copy()
        # Strategy 1: mixing beta
        if new_params["mixing_beta"] > 0.3:
            logger.info("Healing: Reducing mixing_beta to 0.3")
            new_params["mixing_beta"] = 0.3
            return new_params

        # Strategy 2: increase smearing
        if new_params["sigma"] < 0.05:
            logger.info("Healing: Increasing smearing (sigma) to 0.05")
            new_params["sigma"] = 0.05
            return new_params

        # Strategy 3: diagonalization
        if new_params["diagonalization"] == "david":
            logger.info("Healing: Switching diagonalization to cg")
            new_params["diagonalization"] = "cg"
            return new_params

        # Strategy 4: even safer settings?
        if new_params["mixing_beta"] > 0.1:
            logger.info("Healing: Further reducing mixing_beta to 0.1")
            new_params["mixing_beta"] = 0.1
            return new_params

        return new_params
