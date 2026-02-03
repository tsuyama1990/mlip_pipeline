import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ase.io import read, write

from mlip_autopipec.config.config_model import EonConfig
from mlip_autopipec.domain_models.structures import CandidateStructure, StructureMetadata
from mlip_autopipec.inference import pace_driver

logger = logging.getLogger(__name__)


class EonWrapper:
    def __init__(self, config: EonConfig) -> None:
        self.config = config

    def _write_config_ini(self, work_dir: Path) -> None:
        config_path = work_dir / "config.ini"

        # Default EON config
        eon_conf: dict[str, dict[str, Any]] = {
            "Main": {
                "job": "process_search",
                "temperature": 300.0,
                "random_seed": 12345,
            },
            "Potential": {
                "potential": "script",
                "script_path": "./pace_driver.py",
            },
            "Process Search": {
                "minimize_first": "true",
            },
        }

        # Override with user parameters
        for key, value in self.config.parameters.items():
            if isinstance(value, dict):
                # Section override
                if key not in eon_conf:
                    eon_conf[key] = {}
                eon_conf[key].update(value)
            else:
                # Main override
                eon_conf["Main"][key] = value

        with config_path.open("w") as f:
            for section, options in eon_conf.items():
                f.write(f"[{section}]\n")
                for k, v in options.items():
                    f.write(f"{k} = {v}\n")
                f.write("\n")

    def run_akmc(
        self, potential_path: Path | None, structure_path: Path, work_dir: Path
    ) -> list[CandidateStructure]:
        """
        Runs the AKMC search using EON.
        """
        work_dir.mkdir(parents=True, exist_ok=True)

        # 1. Setup config.ini
        self._write_config_ini(work_dir)

        # 2. Copy driver
        driver_src = Path(pace_driver.__file__)
        shutil.copy(driver_src, work_dir / "pace_driver.py")
        (work_dir / "pace_driver.py").chmod(0o755)

        # 3. Copy/Link potential
        if potential_path and potential_path.exists():
            shutil.copy(potential_path, work_dir / "potential.yace")
        else:
            logger.warning("No potential file provided for AKMC run.")

        # 4. Write pos.con
        try:
            structure = read(structure_path)
            # Try writing eon format, fallback to formatting manually if needed
            # ASE 'eon' format usually works.
            write(work_dir / "pos.con", structure, format="eon")
        except Exception:
            logger.exception("Failed to write pos.con")
            # Try generic writer or fail
            # For now, fail
            raise

        # 5. Run EON
        cmd = self.config.command.split()
        logger.info(f"Starting EON in {work_dir}")

        candidates = []

        try:
            # We capture output to avoid spamming logs, but maybe check stderr
            subprocess.run(cmd, cwd=work_dir, check=True, capture_output=True)  # noqa: S603
        except subprocess.CalledProcessError as e:
            if e.returncode == 100:
                logger.info("EON halted due to uncertainty.")
                # If halted, we might want to check if any intermediate files exist
                # But typically driver halt means the current structure (input to driver) was uncertain.
                # EON doesn't save it easily unless we trap it.
                # For cycle 06, we focus on harvesting results or handling successful runs.
            else:
                logger.warning(f"EON execution failed or returned non-zero: {e.stderr.decode()}")
                # Don't crash the pipeline, just return what we have (if anything)
        except FileNotFoundError:
             logger.warning(f"EON command '{self.config.command}' not found. Skipping execution.")
             return []

        # 6. Harvest Candidates
        # Look for process search results
        proc_dir = work_dir / "procdata"
        if proc_dir.exists():
            for product_file in proc_dir.glob("*/product.con"):
                try:
                    cand = CandidateStructure(
                        structure_path=product_file,
                        metadata=StructureMetadata(
                            source="akmc",
                            generation_method="eon_saddle",
                            parent_structure_id=structure_path.name,
                        ),
                    )
                    candidates.append(cand)
                except Exception as e:
                    logger.warning(f"Failed to parse EON output {product_file}: {e}")
                    continue

        logger.info(f"EON AKMC finished. Found {len(candidates)} candidates.")
        return candidates
