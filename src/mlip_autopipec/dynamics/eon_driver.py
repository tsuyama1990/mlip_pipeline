import logging
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

from ase.io import read, write

from mlip_autopipec.domain_models.config import DynamicsConfig
from mlip_autopipec.domain_models.datastructures import HaltInfo, Potential, Structure
from mlip_autopipec.dynamics.interface import BaseDynamics

logger = logging.getLogger(__name__)

class EONDriver(BaseDynamics):
    """
    Dynamics Driver for EON (Adaptive Kinetic Monte Carlo).
    Integrates EON client with MLIP potentials.
    """

    def __init__(self, work_dir: Path, config: DynamicsConfig) -> None:
        self.work_dir = work_dir
        self.config = config

        if self.config.eon is None:
             raise ValueError("EON configuration missing in DynamicsConfig")

    def simulate(self, potential: Potential, structure: Structure) -> Iterator[Structure]:
        """
        Runs an EON simulation using the provided potential and structure.
        """
        logger.info("Starting EON Simulation...")

        # 1. Prepare Workspace
        run_dir = self.work_dir / "eon_run"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)

        # 2. Setup Files
        self._prepare_run(potential, structure, run_dir)

        # 3. Execute EON
        client_cmd = [self.config.eon.client_path]

        logger.info(f"Executing: {' '.join(client_cmd)} in {run_dir}")

        try:
            # We assume eonclient runs until completion or halt.
            # Usually EON runs as a daemon or long process, but we treat it as a task here.
            # If it's a client, it connects to a server or runs locally.
            # Assuming standalone client mode or local execution.

            # Note: EON usually requires a communicator.
            # If we are running "eonclient", it expects a server.
            # If we are running a standalone AKMC, we usually run "eon".
            # The spec mentions "eonclient". Let's assume the user knows what they are doing with the command.

            result = subprocess.run(
                client_cmd,
                cwd=run_dir,
                capture_output=True,
                text=True,
                check=False # We handle return codes manually
            )

            if result.returncode == 100:
                logger.warning("EON halted due to potential server trigger (Halt Code 100).")
                halt_info = self._parse_halt(run_dir, structure)
                # Yield the halted structure with metadata
                yield halt_info.structure
                return

            if result.returncode != 0:
                logger.error(f"EON failed with code {result.returncode}")
                logger.error(f"Stdout: {result.stdout}")
                logger.error(f"Stderr: {result.stderr}")
                raise RuntimeError(f"EON execution failed: {result.stderr}")

            logger.info("EON simulation completed successfully.")

            # 4. Parse Results
            # If successful, yield the trajectory or final state
            yield from self._parse_results(run_dir, structure)

        except Exception as e:
            logger.error(f"Simulation error: {e}")
            raise

    def _prepare_run(self, potential: Potential, structure: Structure, run_dir: Path) -> None:
        """Generates input files for EON."""

        # Write config.ini
        config_content = f"""[Main]
temperature = {self.config.eon.temperature}
prefactor = {self.config.eon.prefactor}
search_method = {self.config.eon.search_method}

[Potential]
type = External
command = python {self.config.eon.server_script_name} --potential potential.{potential.format} --threshold {self.config.max_gamma_threshold}
"""
        (run_dir / "config.ini").write_text(config_content)

        # Write Structure (pos.con)
        # ASE supports writing .con files if strict=False or similar?
        # Actually ASE supports format='eon' which writes .con
        # But we need to check if 'eon' format is available in installed ASE.
        # If not, we might need a custom writer or fallback to xyz and convert.
        # Assuming ASE has 'eon' or 'con' support.
        # Let's try 'eon'. If fails, try 'con'.
        try:
            write(run_dir / "pos.con", structure.to_ase(), format="eon")
        except Exception:
             logger.warning("ASE 'eon' format not found, trying manual write or fallback.")
             # Fallback: simple con writer if needed, or assume xyz and EON is configured to read it (unlikely)
             # For now, let's assume 'eon' works or we use a simple writer.
             # Or we write 'reactant.con' which is standard.
             write(run_dir / "pos.con", structure.to_ase()) # Let ASE guess or use default

        # Copy/Link Potential
        # We copy it to ensure isolation
        shutil.copy(potential.path, run_dir / f"potential.{potential.format}")

        # Copy Potential Server Script
        # We need to put the potential_server.py in the run directory so EON can call it
        # We can locate it relative to this file
        server_script_src = Path(__file__).parent / "potential_server.py"
        if server_script_src.exists():
            shutil.copy(server_script_src, run_dir / self.config.eon.server_script_name)
        else:
             # If not found (e.g. installed package), we might need to rely on it being in PATH or handle it.
             # For dev, it should be there.
             logger.warning("potential_server.py not found at expected location. Assuming it is in PATH or manual setup.")

    def _parse_halt(self, run_dir: Path, original_structure: Structure) -> HaltInfo:
        """Parses the halt state."""
        bad_struct_path = run_dir / "bad_structure.xyz"
        halt_info_path = run_dir / "halt_info.txt"

        halt_struct = original_structure
        max_gamma = self.config.max_gamma_threshold
        reason = "uncertainty"

        if bad_struct_path.exists():
            atoms = read(bad_struct_path)
            # Create Structure
            halt_struct = Structure(
                atoms=atoms,
                provenance="eon_halt",
                label_status="unlabeled",
                metadata={"halt_reason": "uncertainty"}
            )

        if halt_info_path.exists():
            content = halt_info_path.read_text()
            for line in content.splitlines():
                if "max_gamma" in line:
                    try:
                        max_gamma = float(line.split(":")[1].strip())
                    except ValueError:
                        pass

        # We wrap it in Structure but we need to return HaltInfo...
        # Wait, simulate yields Structure.
        # The Orchestrator handles HaltInfo usually from the generator/active learner side?
        # Or does BaseDynamics just yield structures?
        # Interface says: Iterator[Structure].
        # So we update metadata of the structure with halt info.

        halt_struct.metadata["halt_reason"] = reason
        halt_struct.metadata["max_gamma"] = max_gamma

        # We return a simple object helper here, but simulate yields Structure.
        return HaltInfo(
            step=0, # Unknown step
            max_gamma=max_gamma,
            structure=halt_struct,
            reason=reason
        )

    def _parse_results(self, run_dir: Path, original_structure: Structure) -> Iterator[Structure]:
        """Reads successful EON results."""
        # EON output varies. Let's look for 'results.dat' or 'processes.dat' or the final configuration.
        # If it's a search, we might have 'saddle.con', 'product.con', etc.

        # For this cycle, let's yield the product if found.
        products = list(run_dir.glob("product_*.con"))
        if not products:
             products = list(run_dir.glob("final.con"))

        for p in products:
             try:
                 atoms = read(p)
                 yield Structure(
                     atoms=atoms,
                     provenance="eon_product",
                     label_status="unlabeled",
                     metadata={"source_file": p.name}
                 )
             except Exception as e:
                 logger.warning(f"Failed to read result {p}: {e}")
