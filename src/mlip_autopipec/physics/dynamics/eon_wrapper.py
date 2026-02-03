import configparser
import logging
import shutil
import subprocess
from pathlib import Path

from ase import Atoms
from ase.io import write

from mlip_autopipec.config.config_model import Config
from mlip_autopipec.inference import pace_driver

logger = logging.getLogger(__name__)


class EonWrapper:
    def __init__(self, config: Config) -> None:
        self.config = config

    def run_akmc(self, potential_path: Path, structure: Atoms, work_dir: Path) -> int:
        """
        Sets up and runs EON AKMC simulation.
        Returns the exit code of the eon process.
        If the driver halts with 100, EON should exit with error or status.
        We check the return code.
        """
        work_dir.mkdir(parents=True, exist_ok=True)

        # 1. Setup EON directory
        self._write_config_ini(work_dir)
        self._write_pos_con(structure, work_dir)
        self._install_driver(work_dir)
        self._link_potential(potential_path, work_dir)

        # 2. Run EON
        # Default command "eonclient"
        eon_cmd_str = "eonclient"
        if self.config.eon:
            eon_cmd_str = self.config.eon.command

        logger.info(f"Starting EON in {work_dir}")
        try:
            # Check if command exists (optional, but good practice)
            # shutil.which might not find it if it's an absolute path or alias,
            # but usually fine.
            cmd_parts = eon_cmd_str.split()
            if not shutil.which(cmd_parts[0]) and not Path(cmd_parts[0]).exists():
                logger.warning(f"EON command '{cmd_parts[0]}' not found in PATH.")

            # Run EON
            with (work_dir / "eon.log").open("w") as log_file:
                # We assume eonclient runs in foreground and exits when done or error
                process = subprocess.run(  # noqa: S603
                    cmd_parts,
                    cwd=work_dir,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=False,
                )

        except Exception:
            logger.exception("EON execution failed")
            return 1
        else:
            if process.returncode != 0:
                logger.warning(f"EON exited with code {process.returncode}")

            return process.returncode

    def _write_config_ini(self, work_dir: Path) -> None:
        config = configparser.ConfigParser()
        config.optionxform = str  # type: ignore # Preserve case

        # Default EON settings
        config["Main"] = {
            "job": "process_search",
            "temperature": "300.0",
            "random_seed": "0",
            "potential": "python3 pace_driver.py",
        }

        if self.config.eon and self.config.eon.parameters:
            for key, value in self.config.eon.parameters.items():
                if isinstance(value, dict):
                    if key not in config:
                        config[key] = {}
                    for k, v in value.items():
                        config[key][str(k)] = str(v)
                else:
                    config["Main"][key] = str(value)

        with (work_dir / "config.ini").open("w") as f:
            config.write(f)

    def _write_pos_con(self, structure: Atoms, work_dir: Path) -> None:
        write(work_dir / "pos.con", structure, format="eon")

    def _install_driver(self, work_dir: Path) -> None:
        src_file = Path(pace_driver.__file__)
        shutil.copy(src_file, work_dir / "pace_driver.py")

    def _link_potential(self, potential_path: Path, work_dir: Path) -> None:
        dest = work_dir / "potential.yace"
        if dest.exists():
            dest.unlink()
        # Using copy instead of symlink for robustness/portability
        shutil.copy(potential_path, dest)
