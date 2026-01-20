import shutil
import subprocess
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from ase import Atoms

from mlip_autopipec.core.config import DFTConfig
from mlip_autopipec.data_models.dft_models import DFTResult
from mlip_autopipec.dft.input_gen import write_pw_input
from mlip_autopipec.dft.parsers import parse_pw_output
from mlip_autopipec.dft.utils import get_kpoints, get_sssp_pseudopotentials, is_magnetic
from mlip_autopipec.exceptions import DFTCalculationException, DFTException


class QERunner:
    def __init__(self, config: DFTConfig) -> None:
        self.config = config

    def _prepare_parameters(
        self, atoms: Atoms
    ) -> tuple[dict[str, Any], dict[str, str], list[int]]:
        """
        Prepares DFT parameters, pseudopotentials, and k-points.
        """
        pseudos = get_sssp_pseudopotentials(atoms, self.config.pseudopotential_dir)
        kpoints = get_kpoints(atoms, density=0.15)

        params: dict[str, Any] = {}
        # Base parameters
        params["mixing_beta"] = self.config.mixing_beta
        params["k_points"] = kpoints

        system_params: dict[str, Any] = {}
        system_params["ecutwfc"] = 50.0  # Default Ry
        system_params["ecutrho"] = 200.0
        system_params["occupations"] = "smearing"
        system_params["smearing"] = self.config.smearing
        system_params["degauss"] = 0.02

        if is_magnetic(atoms):
            system_params["nspin"] = 2
            # Set simple initial moments if not present
            moms = atoms.get_initial_magnetic_moments()
            if not moms.any():
                atoms.set_initial_magnetic_moments([1.0] * len(atoms))

        params["system"] = system_params
        return params, pseudos, kpoints

    def run_static_calculation(self, atoms: Atoms, run_dir: Path) -> DFTResult:
        """
        Executes a static DFT calculation.
        """
        run_dir.mkdir(parents=True, exist_ok=True)
        job_id = str(uuid4())

        params, pseudos, kpoints = self._prepare_parameters(atoms)

        # 3. Generate Input
        # Cast kpoints to tuple for write_pw_input
        if len(kpoints) != 3:
            msg = "kpoints must have 3 components"
            raise DFTException(msg)
        kpts_tuple = (kpoints[0], kpoints[1], kpoints[2])
        input_str = write_pw_input(atoms, params, pseudos, kpts_tuple)
        input_path = run_dir / "pw.in"
        input_path.write_text(input_str)

        output_path = run_dir / "pw.out"

        # 4. Execute
        cmd_list = self.config.command.split()
        full_cmd = [*cmd_list, "-in", str(input_path.name)]

        start_time = time.time()
        try:
            with output_path.open("w") as f_out:
                # subprocess.run waits for the process to complete.
                # If timeout occurs, it kills the child process and raises TimeoutExpired.
                subprocess.run(  # noqa: S603
                    full_cmd,
                    cwd=str(run_dir),
                    stdout=f_out,
                    stderr=subprocess.PIPE,
                    check=True,
                    timeout=3600,  # 1 hour hard limit
                )
        except subprocess.TimeoutExpired as e:
            msg = "DFT calculation timed out"
            # The process is already killed by subprocess.run on timeout
            raise DFTCalculationException(msg, is_timeout=True) from e
        except subprocess.CalledProcessError as e:
            stderr_output = e.stderr.decode() if e.stderr else ""
            msg = f"DFT execution failed: {stderr_output}"
            raise DFTCalculationException(msg, stderr=stderr_output) from e
        except Exception as e:
            msg = f"DFT execution error: {e}"
            raise DFTException(msg) from e

        wall_time = time.time() - start_time

        # 5. Parse
        result = parse_pw_output(output_path, job_id, wall_time, params)

        # 6. Clean
        try:
            for ext in [".wfc", ".hub", ".mix", ".save"]:
                for f in run_dir.glob(f"*{ext}"):
                    if f.is_dir():
                        shutil.rmtree(f, ignore_errors=True)
                    else:
                        f.unlink(missing_ok=True)
        except OSError:
            # Non-critical cleanup failure
            pass

        return result
