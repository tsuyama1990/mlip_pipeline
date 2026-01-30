import logging
import subprocess
from pathlib import Path

import ase.io

from mlip_autopipec.domain_models.config import LammpsConfig, MDParams, PotentialConfig
from mlip_autopipec.domain_models.job import JobStatus, LammpsResult
from mlip_autopipec.domain_models.structure import Structure

logger = logging.getLogger(__name__)


class LammpsRunner:
    """Wrapper for running LAMMPS simulations."""

    def __init__(self, config: LammpsConfig) -> None:
        self.config = config

    def run(self, structure: Structure, params: MDParams, potential_config: PotentialConfig, work_dir: Path) -> LammpsResult:
        """
        Run a LAMMPS simulation.

        Args:
            structure: Initial structure.
            params: MD parameters.
            potential_config: Potential configuration including element parameters.
            work_dir: Directory to run simulation in.

        Returns:
            LammpsResult object.
        """
        work_dir.mkdir(parents=True, exist_ok=True)
        job_id = work_dir.name

        try:
            # 1. Write Inputs
            self._write_inputs(work_dir, structure, params, potential_config)

            # 2. Run LAMMPS
            command = self.config.command.split()
            command.extend(["-in", "in.lammps"])

            logger.info(f"Running LAMMPS in {work_dir}: {' '.join(command)}")

            # Use a log file
            log_file = work_dir / "log.lammps"

            # Run subprocess
            try:
                result = subprocess.run(
                    command,
                    cwd=work_dir,
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout
                )

                # Write stdout/stderr to log
                with log_file.open("w") as f:
                    f.write(result.stdout)
                    f.write("\nSTDERR:\n")
                    f.write(result.stderr)

                if result.returncode != 0:
                    return LammpsResult(
                        job_id=job_id,
                        status=JobStatus.FAILED,
                        work_dir=work_dir,
                        duration_seconds=0.0,
                        log_content=result.stderr[-500:],
                        final_structure=structure, # Return initial as fallback
                        trajectory_path=work_dir / "dump.lammpstrj"
                    )

            except subprocess.TimeoutExpired:
                logger.error(f"Job {job_id} timed out after {self.config.timeout}s")
                return LammpsResult(
                    job_id=job_id,
                    status=JobStatus.TIMEOUT,
                    work_dir=work_dir,
                    duration_seconds=float(self.config.timeout),
                    log_content="Timeout Expired",
                    final_structure=structure,
                    trajectory_path=work_dir / "dump.lammpstrj"
                )

            # 3. Parse Output
            final_structure = self._parse_output(work_dir / "dump.lammpstrj")

            # Rough duration (placeholder)
            duration = 0.0 # TODO: Calculate properly

            return LammpsResult(
                job_id=job_id,
                status=JobStatus.COMPLETED,
                work_dir=work_dir,
                duration_seconds=duration,
                log_content=result.stdout[-500:],
                final_structure=final_structure,
                trajectory_path=work_dir / "dump.lammpstrj"
            )

        except Exception as e:
            logger.exception(f"Unexpected error in LammpsRunner: {e}")
            return LammpsResult(
                job_id=job_id,
                status=JobStatus.FAILED,
                work_dir=work_dir,
                duration_seconds=0.0,
                log_content=str(e),
                final_structure=structure,
                trajectory_path=work_dir / "dump.lammpstrj"
            )

    def _write_inputs(self, work_dir: Path, structure: Structure, params: MDParams, potential_config: PotentialConfig) -> None:
        """Write LAMMPS data and input script."""
        atoms = structure.to_ase()

        # Write data file
        ase.io.write(work_dir / "data.lammps", atoms, format="lammps-data") # type: ignore[no-untyped-call]

        # Get unique elements present in structure to ensure we have params
        present_elements = set(structure.symbols)

        # Generate mass and pair_coeff commands
        mass_cmds = []
        pair_coeff_lj = []
        pair_coeff_zbl = []

        # NOTE: ase.io.write lammps-data writes types as 1, 2, 3... corresponding to sorted species?
        # Typically ASE sorts them alphabetically or by appearance.
        # Ideally we should strictly map types.
        # But 'lammps-data' output usually follows alphabetical order of symbols if created from ASE.
        # We need to rely on ASE's mapping.

        # Let's assume sorted symbols list
        sorted_elements = sorted(list(present_elements))

        for i, el in enumerate(sorted_elements, start=1):
            if el not in potential_config.element_params:
                raise ValueError(f"Missing element parameters for {el}")

            p = potential_config.element_params[el]

            mass_cmds.append(f"mass            {i} {p.mass}")

            # LJ: pair_coeff type1 type2 epsilon sigma cutoff
            # Here we do diagonal only for simplicity in loop, but LAMMPS needs all pairs or mixing.
            # "pair_coeff * * ..." sets global or specific.
            # For hybrid, we usually specify specific types.

            # Simplified: Use * * for single element system or simple mixing
            # If multi-element, this logic needs to be pair-wise.
            # SPEC says "programmatically".

            # For this cycle (Si only implied but code should be generic):
            # pair_coeff {i} {i} ...

            pair_coeff_lj.append(f"pair_coeff      {i} {i} lj/cut {p.lj_epsilon} {p.lj_sigma}")

            # ZBL: pair_coeff type1 type2 Zi Zj
            pair_coeff_zbl.append(f"pair_coeff      {i} {i} zbl {p.zbl_z} {p.zbl_z}")

        # If we have mixed interactions, we rely on mixing rules or explicit defs.
        # For One-Shot MD with single element, this loop works.
        # For multi-element, we'd need cross terms.
        # Sticking to diagonal for now as Cycle 2 focus is basic exploration.

        script = f"""
units           metal
atom_style      atomic
boundary        p p p

read_data       data.lammps

# Hybrid pair style: LJ/Cut for baseline + ZBL
pair_style      hybrid/overlay lj/cut 2.5 zbl 2.0 2.5

{''.join(mass_cmds)}

{''.join(pair_coeff_lj)}
{''.join(pair_coeff_zbl)}

velocity        all create {params.temperature} 12345 mom yes rot no

fix             1 all {params.ensemble} temp {params.temperature} {params.temperature} 0.1
"""
        if params.ensemble == "npt" and params.pressure is not None:
             script = script.replace(f"fix             1 all {params.ensemble}", f"fix             1 all npt temp {params.temperature} {params.temperature} 0.1 iso {params.pressure} {params.pressure} 1.0")

        script += f"""
timestep        {params.timestep}
thermo          100

dump            1 all custom 100 dump.lammpstrj id type x y z
run             {params.n_steps}
"""
        (work_dir / "in.lammps").write_text(script)

    def _parse_output(self, dump_path: Path) -> Structure:
        """Parse the final frame from LAMMPS dump using streaming read."""
        if not dump_path.exists():
            raise FileNotFoundError(f"Dump file not found: {dump_path}")

        # Use ASE iread to avoid loading full trajectory into memory
        last_atoms = None
        # type: ignore[no-untyped-call]
        for atoms in ase.io.iread(dump_path, format="lammps-dump-text"):
            last_atoms = atoms

        if last_atoms is None:
             raise ValueError("Dump file is empty")

        return Structure.from_ase(last_atoms) # type: ignore[arg-type]
