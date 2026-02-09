import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read, write
from ase.units import bar

from mlip_autopipec.components.dynamics.hybrid import generate_pair_style
from mlip_autopipec.domain_models.config import PhysicsBaselineConfig
from mlip_autopipec.domain_models.potential import Potential

logger = logging.getLogger(__name__)


class LammpsSinglePointCalculator(Calculator):
    """
    ASE Calculator that uses LAMMPS to compute energy, forces, and stress
    for a single configuration (static calculation).
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        potential: Potential,
        workdir: Path,
        command: str = "lmp",
        physics_baseline: dict[str, Any] | None = None,
        keep_files: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.potential = potential
        self.workdir = workdir
        self.command = command
        self.physics_baseline = physics_baseline
        self.keep_files = keep_files
        self.workdir.mkdir(parents=True, exist_ok=True)

    def calculate(
        self,
        atoms: Atoms = None,
        properties: list[str] = ["energy"],
        system_changes: list[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)

        # 1. Write data file
        data_file = self.workdir / "data.lammps"
        try:
            write(data_file, self.atoms, format="lammps-data", atom_style="atomic")
        except Exception:
             write(data_file, self.atoms, format="lammps-data")

        # 2. Write input file
        input_file = self.workdir / "in.lammps"
        log_file = self.workdir / "log.lammps"
        dump_file = self.workdir / "dump.lammps"

        baseline_config = None
        if self.physics_baseline:
            baseline_config = PhysicsBaselineConfig.model_validate(self.physics_baseline)

        pair_style, pair_coeff = generate_pair_style(self.potential, baseline_config)

        input_content = f"""
units           metal
atom_style      atomic
boundary        p p p

read_data       {data_file.name}

{pair_style}
{pair_coeff}

# Compute forces and stress
compute         stress all pressure thermo_temp
variable        pxx equal c_stress[1]
variable        pyy equal c_stress[2]
variable        pzz equal c_stress[3]
variable        pyz equal c_stress[4]
variable        pxz equal c_stress[5]
variable        pxy equal c_stress[6]

thermo_style    custom step temp pe etotal press v_pxx v_pyy v_pzz v_pyz v_pxz v_pxy

dump            1 all custom 1 {dump_file.name} id type x y z fx fy fz

run             0
"""
        input_file.write_text(input_content)

        # 3. Run LAMMPS
        if not shutil.which(self.command):
            logger.warning(f"LAMMPS binary '{self.command}' not found. Cannot run calculation.")
            # We raise so upper layers catch it
            msg = f"LAMMPS binary '{self.command}' not found."
            raise RuntimeError(msg)

        try:
            cmd = [self.command, "-in", str(input_file.name), "-log", str(log_file.name)]
            subprocess.run(
                cmd, cwd=self.workdir, capture_output=True, text=True, check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            log_content = log_file.read_text() if log_file.exists() else "No log"
            msg = f"LAMMPS failed: {e}\nLog: {log_content}"
            logger.error(msg)
            raise RuntimeError(msg) from e

        # 4. Read results
        self.results = {}
        self._read_log(log_file)
        self._read_dump(dump_file)

        if not self.keep_files:
            pass

    def _read_log(self, log_file: Path) -> None:
        content = log_file.read_text()
        lines = content.splitlines()

        found = False
        for line in reversed(lines):
            parts = line.split()
            if parts and parts[0] == "0" and len(parts) >= 11:
                try:
                    pe = float(parts[2])
                    self.results["energy"] = pe

                    stress_voigt = [
                        -float(parts[5]) * bar, # xx
                        -float(parts[6]) * bar, # yy
                        -float(parts[7]) * bar, # zz
                        -float(parts[8]) * bar, # yz
                        -float(parts[9]) * bar, # xz
                        -float(parts[10]) * bar # xy
                    ]
                    self.results["stress"] = stress_voigt
                    found = True
                    break
                except ValueError:
                    continue

        if not found:
            raise RuntimeError("Could not find thermo output in log")

    def _read_dump(self, dump_file: Path) -> None:
        try:
            atoms_list = read(dump_file, index=":", format="lammps-dump-text")
            if not atoms_list:
                msg = "No atoms found in dump"
                raise RuntimeError(msg)

            atoms = atoms_list[-1]
            forces = atoms.get_forces() # type: ignore[no-untyped-call]
            self.results["forces"] = forces

        except Exception as e:
            msg = f"Failed to read dump file: {e}"
            raise RuntimeError(msg) from e
