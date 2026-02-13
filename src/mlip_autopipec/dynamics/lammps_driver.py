import logging
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from ase.data import atomic_numbers
from ase.io import iread, write

from mlip_autopipec.domain_models.config import DynamicsConfig
from mlip_autopipec.domain_models.datastructures import Potential, Structure
from mlip_autopipec.dynamics.hybrid_overlay import HybridOverlay
from mlip_autopipec.dynamics.interface import BaseDynamics
from mlip_autopipec.dynamics.watchdog import UncertaintyWatchdog

logger = logging.getLogger(__name__)


class LAMMPSDriver(BaseDynamics):
    """LAMMPS implementation of Dynamics Engine."""

    def __init__(self, work_dir: Path, config: DynamicsConfig) -> None:
        self.work_dir = work_dir
        self.config = config
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def simulate(self, potential: Potential, structure: Structure) -> Iterator[Structure]:
        """
        Runs a LAMMPS simulation using the potential.
        """
        logger.info("LAMMPSDriver: Setting up simulation...")

        run_dir = self._setup_run_directory(structure)
        # We need elements to correctly map types back when parsing the dump file
        ase_atoms = structure.to_ase()
        elements = sorted(set(ase_atoms.get_chemical_symbols()))  # type: ignore[no-untyped-call]

        _, _, dump_file = self._prepare_simulation_files(run_dir, structure, potential, elements)

        self._run_lammps(run_dir)

        yield from self._parse_dump_file(dump_file, elements)

    def _setup_run_directory(self, structure: Structure) -> Path:
        run_name = f"md_run_{structure.provenance}" if structure.provenance else "md_run"
        if len(run_name) > 50:
            run_name = run_name[:50]

        run_dir = self.work_dir / run_name
        run_dir.mkdir(exist_ok=True)
        return run_dir

    def _prepare_simulation_files(
        self, run_dir: Path, structure: Structure, potential: Potential, elements: list[str]
    ) -> tuple[Path, Path, Path]:
        data_file = run_dir / "structure.data"
        input_file = run_dir / "in.md"
        potential_link = run_dir / "potential.yace"
        dump_file = run_dir / "traj.dump"

        # Link/Copy potential
        if potential.path.exists():
            shutil.copy(potential.path, potential_link)
        else:
            msg = f"Potential not found: {potential.path}"
            raise FileNotFoundError(msg)

        # Write structure
        ase_atoms = structure.to_ase()
        write(data_file, ase_atoms, format="lammps-data", specorder=elements)

        # Generate Input Script
        commands = self._generate_input_script(
            structure_file="structure.data",
            potential_file="potential.yace",
            dump_file="traj.dump",
            elements=elements,
        )
        input_file.write_text(commands)

        return data_file, input_file, dump_file

    def _run_lammps(self, run_dir: Path) -> None:
        logger.info(f"LAMMPSDriver: Executing in {run_dir}")
        # Command is hardcoded safe list, no shell=True
        cmd = ["lmp", "-in", "in.md"]

        try:
            with (run_dir / "stdout.log").open("w") as stdout:
                # Use check=True to raise CalledProcessError on non-zero exit
                # We catch it to handle special code 100
                subprocess.run(  # noqa: S603
                    cmd, cwd=run_dir, stdout=stdout, stderr=subprocess.STDOUT, check=True
                )
        except subprocess.CalledProcessError as e:
            if e.returncode == 100:
                logger.warning("LAMMPSDriver: Simulation halted by Watchdog (Code 100).")
            else:
                logger.exception(f"LAMMPSDriver: Simulation failed with code {e.returncode}")
                # We continue to try parsing what we have, as partial trajectory might be useful
        except FileNotFoundError as err:
            logger.exception("LAMMPS executable 'lmp' not found in PATH.")
            msg = "LAMMPS executable not found."
            raise RuntimeError(msg) from err

    def _parse_dump_file(self, dump_file: Path, elements: list[str]) -> Iterator[Structure]:
        # Use explicit frame limit to prevent OOM
        max_frames = 10000
        if dump_file.exists():
            # Use 'index=":"' to create a generator, preventing full file read
            for i, atoms in enumerate(iread(dump_file, format="lammps-dump-text", index=":")):
                if i >= max_frames:
                    logger.warning(f"LAMMPSDriver: Reached max frame limit ({max_frames}).")
                    break

                types = atoms.get_atomic_numbers()  # type: ignore[no-untyped-call]
                real_numbers = [atomic_numbers[elements[t - 1]] for t in types]
                atoms.set_atomic_numbers(real_numbers)  # type: ignore[no-untyped-call]

                gamma = 0.0
                if "c_pace[1]" in atoms.arrays:
                    gammas = atoms.arrays["c_pace[1]"]
                    if len(gammas) > 0:
                        gamma = np.max(gammas)

                yield Structure(
                    atoms=atoms,
                    provenance="md_trajectory",
                    label_status="unlabeled",
                    uncertainty_score=float(gamma),
                    metadata={"temperature": self.config.temperature, "frame": i},
                )
        else:
            logger.warning("LAMMPSDriver: No dump file produced.")

    # Correcting implementation to pass elements

    def _generate_input_script(
        self, structure_file: str, potential_file: str, dump_file: str, elements: list[str]
    ) -> str:
        hybrid = HybridOverlay(self.config)
        watchdog = UncertaintyWatchdog(self.config)

        lines = [
            "units metal",
            "atom_style atomic",
            "boundary p p p",
            "read_data " + structure_file,
            "mass * 1.0",
            hybrid.get_pair_style(),
            hybrid.get_pair_coeff(elements, potential_file),
            "neighbor 1.0 bin",
            "neigh_modify delay 0 every 1 check yes",
        ]

        wd_cmds = watchdog.get_commands(potential_file, elements)
        if wd_cmds:
            lines.append(wd_cmds)

        lines.append(f"timestep {self.config.timestep}")
        lines.append(
            f"velocity all create {self.config.temperature} 12345 mom yes rot yes dist gaussian"
        )
        lines.append(
            f"fix 1 all nvt temp {self.config.temperature} {self.config.temperature} $(100.0*dt)"
        )

        thermo_args = "step temp press etotal"
        dump_args = "id type x y z"

        if self.config.halt_on_uncertainty:
            thermo_args += " c_max_gamma"
            dump_args += " c_pace[1]"

        lines.append(f"thermo {self.config.n_thermo}")
        lines.append(f"thermo_style custom {thermo_args}")

        lines.append(f"dump 1 all custom {self.config.n_dump} {dump_file} {dump_args}")
        lines.append("dump_modify 1 sort id")

        lines.append(f"run {self.config.steps}")

        return "\n".join(lines)
