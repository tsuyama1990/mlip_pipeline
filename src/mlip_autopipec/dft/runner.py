import subprocess
import shutil
import shlex
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from ase import Atoms
from ase.io import write, read
from ase.stress import voigt_6_to_full_3x3_stress
import numpy as np

from mlip_autopipec.config.schemas.dft import DFTConfig
from mlip_autopipec.data_models.dft_models import DFTResult

class QERunner:
    def __init__(self, config: DFTConfig, work_dir: Path):
        self.config = config
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def run(self, atoms: Atoms) -> DFTResult:
        # 1. Write Input
        input_path = self.work_dir / "pw.in"
        output_path = self.work_dir / "pw.out"

        try:
            self._write_input(atoms, input_path)
        except Exception as e:
            return DFTResult(
                energy=0.0,
                forces=[],
                converged=False,
                error_message=f"Input generation failed: {e}"
            )

        # 2. Run Command
        success, error_msg = self._run_command(input_path, output_path)

        if not success:
            return DFTResult(
                energy=0.0,
                forces=[],
                converged=False,
                error_message=error_msg
            )

        # 3. Parse Output
        return self._parse_output(output_path)

    def _write_input(self, atoms: Atoms, path: Path) -> None:
        input_data: Dict[str, Any] = {
            'control': {
                'calculation': 'scf',
                'restart_mode': 'from_scratch',
                'tprnfor': True,
                'tstress': True,
                'pseudo_dir': str(self.config.pseudopotential_dir),
                'disk_io': 'none',
            },
            'system': {
                'ecutwfc': getattr(self.config, 'ecutwfc', 40.0),
                'occupations': 'smearing',
                'smearing': getattr(self.config, 'smearing', 'mv'),
                'degauss': getattr(self.config, 'degauss', 0.01),
            },
            'electrons': {
                'conv_thr': 1.0e-6,
            }
        }

        if self.config.scf_params:
            for section, values in self.config.scf_params.items():
                if section in input_data and isinstance(values, dict) and isinstance(input_data[section], dict):
                    input_data[section].update(values)
                else:
                    input_data[section] = values

        # type: ignore[no-untyped-call]
        write(
            path,
            atoms,
            format='espresso-in',
            input_data=input_data,
            pseudopotentials=self.config.pseudopotentials,
            kspacing=self.config.kspacing
        )

    def _run_command(self, input_path: Path, output_path: Path) -> Tuple[bool, str]:
        if not self.config.command:
             return False, "Command is empty"

        # Security check (redundant if config validates, but good practice)
        if any(c in self.config.command for c in [";", "&", "|"]):
             return False, "Unsafe characters in command"

        cmd_parts = shlex.split(self.config.command)
        if not cmd_parts:
             return False, "Command parses to empty list"

        executable = cmd_parts[0]
        if not shutil.which(executable):
             return False, f"Executable {executable} not found in PATH"

        try:
            with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
                # noqa: S603 - Command is validated
                result = subprocess.run(
                    cmd_parts,
                    stdin=fin,
                    stdout=fout,
                    stderr=subprocess.PIPE,
                    check=False,
                    shell=False,
                    cwd=self.work_dir
                )

            if result.returncode != 0:
                err = result.stderr.decode()
                return False, f"Return code {result.returncode}: {err}"

            return True, ""

        except Exception as e:
            return False, str(e)

    def _parse_output(self, output_path: Path) -> DFTResult:
        try:
            # type: ignore[no-untyped-call]
            atoms_list = read(output_path, index=':', format='espresso-out')

            if not atoms_list or not isinstance(atoms_list, list):
                 return DFTResult(
                     energy=0.0,
                     forces=[],
                     converged=False,
                     error_message="No output parsed"
                 )

            atoms = atoms_list[-1]
            if not isinstance(atoms, Atoms):
                return DFTResult(
                     energy=0.0,
                     forces=[],
                     converged=False,
                     error_message="Invalid atoms object parsed"
                 )

            energy = atoms.get_potential_energy()
            forces = atoms.get_forces().tolist()

            stress: Optional[List[List[float]]] = None
            raw_stress = atoms.get_stress()
            if raw_stress is not None:
                if len(raw_stress) == 6:
                    stress = voigt_6_to_full_3x3_stress(raw_stress).tolist()
                else:
                    stress = raw_stress.tolist()

            return DFTResult(
                energy=float(energy),
                forces=forces,
                stress=stress,
                converged=True
            )

        except Exception as e:
            return DFTResult(
                energy=0.0,
                forces=[],
                converged=False,
                error_message=f"Parsing failed: {str(e)}"
            )
