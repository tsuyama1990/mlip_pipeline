import contextlib
import logging
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

from ase.io import read, write

from mlip_autopipec.domain_models.config import DynamicsConfig
from mlip_autopipec.domain_models.datastructures import (
    HaltInfo,
    Potential,
    Structure,
    validate_path_safety,
)
from mlip_autopipec.dynamics.interface import BaseDynamics

logger = logging.getLogger(__name__)

class EONExecutionError(RuntimeError):
    """Raised when EON client execution fails."""

class EONDriver(BaseDynamics):
    """
    Dynamics Driver for EON (Adaptive Kinetic Monte Carlo).
    Integrates EON client with MLIP potentials.
    """

    def __init__(self, work_dir: Path, config: DynamicsConfig) -> None:
        self.work_dir = work_dir
        self.config = config

        if self.config.eon is None:
             msg = "EON configuration missing in DynamicsConfig"
             raise ValueError(msg)
        self.eon_config = self.config.eon

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
        client_cmd = [self.eon_config.client_path]

        logger.info("Executing: %s in %s", ' '.join(client_cmd), run_dir)

        try:
            # We assume the user has configured client_path securely in config.
            # S603 check ignored as this is a configured command execution driver.
            result = subprocess.run(  # noqa: S603
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
                logger.error("EON failed with code %s", result.returncode)
                logger.error("Stdout: %s", result.stdout)
                logger.error("Stderr: %s", result.stderr)
                self._raise_runtime_error(result.stderr)

            logger.info("EON simulation completed successfully.")

            # 4. Parse Results
            yield from self._parse_results(run_dir)

        except Exception:
            logger.exception("Simulation error")
            raise

    def _raise_runtime_error(self, stderr: str) -> None:
        """Helper to raise exception with clean message."""
        msg = f"EON execution failed: {stderr}"
        raise EONExecutionError(msg)

    def _prepare_run(self, potential: Potential, structure: Structure, run_dir: Path) -> None:
        """Generates input files for EON."""

        # Validate paths to prevent traversal if they were user inputs (potential path is validated in model, structure provenance is safe)
        # But we double check relative paths constructed here.
        validate_path_safety(run_dir)

        # Write config.ini using template
        config_content = self.eon_config.config_template.format(
            temperature=self.eon_config.temperature,
            prefactor=self.eon_config.prefactor,
            search_method=self.eon_config.search_method,
            server_script=self.eon_config.server_script_name,
            potential_file=f"potential.{potential.format}",
            threshold=self.config.max_gamma_threshold
        )
        (run_dir / "config.ini").write_text(config_content)

        # Write Structure (pos.con)
        try:
            write(run_dir / "pos.con", structure.to_ase(), format="eon") # type: ignore[no-untyped-call]
        except Exception:
             logger.warning("ASE 'eon' format not found, trying manual write or fallback.")
             write(run_dir / "pos.con", structure.to_ase()) # type: ignore[no-untyped-call]

        # Copy/Link Potential
        shutil.copy(potential.path, run_dir / f"potential.{potential.format}")

        # Copy Potential Server Script
        server_script_src = Path(__file__).parent / "potential_server.py"
        if server_script_src.exists():
            shutil.copy(server_script_src, run_dir / self.eon_config.server_script_name)
        else:
             logger.warning("potential_server.py not found at expected location.")

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
                atoms=atoms, # type: ignore[arg-type]
                provenance="eon_halt",
                label_status="unlabeled",
                metadata={"halt_reason": "uncertainty"}
            )

        if halt_info_path.exists():
            content = halt_info_path.read_text()
            for line in content.splitlines():
                if "max_gamma" in line:
                    with contextlib.suppress(ValueError):
                        max_gamma = float(line.split(":")[1].strip())

        halt_struct.metadata["halt_reason"] = reason
        halt_struct.metadata["max_gamma"] = max_gamma

        return HaltInfo(
            step=0, # Unknown step
            max_gamma=max_gamma,
            structure=halt_struct,
            reason=reason
        )

    def _parse_results(self, run_dir: Path) -> Iterator[Structure]:
        """Reads successful EON results using streaming."""
        # Use iterator for memory efficiency and limit results
        # glob returns iterator in Python 3.10+
        products = run_dir.glob("product_*.con")

        found = False
        count = 0
        limit = self.eon_config.max_result_files

        for p in products:
             if count >= limit:
                 logger.warning("Max result files limit reached (%d). Stopping iteration.", limit)
                 break

             found = True
             try:
                 atoms = read(p)
                 yield Structure(
                     atoms=atoms, # type: ignore[arg-type]
                     provenance="eon_product",
                     label_status="unlabeled",
                     metadata={"source_file": p.name}
                 )
                 count += 1
             except Exception:
                 logger.warning("Failed to read result %s", p)

        if not found:
             # Try final.con
             final = run_dir / "final.con"
             if final.exists():
                 try:
                     atoms = read(final)
                     yield Structure(
                         atoms=atoms, # type: ignore[arg-type]
                         provenance="eon_product",
                         label_status="unlabeled",
                         metadata={"source_file": "final.con"}
                     )
                 except Exception:
                     logger.warning("Failed to read final.con")
