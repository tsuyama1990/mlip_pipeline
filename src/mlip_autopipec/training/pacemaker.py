import logging
import shutil
import subprocess
from pathlib import Path

from mlip_autopipec.config.schemas.training import TrainingConfig
from mlip_autopipec.training.config_gen import PacemakerConfigGenerator

logger = logging.getLogger(__name__)


class TrainingResult:
    def __init__(
        self, success: bool, potential_path: Path | None = None, metrics: dict | None = None
    ):
        self.success = success
        self.potential_path = potential_path
        self.metrics = metrics or {}


class PacemakerWrapper:
    """
    Wrapper for executing Pacemaker training commands.
    """

    def __init__(self, config: TrainingConfig, work_dir: Path):
        self.config = config
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def train(self, initial_potential: Path | None = None) -> TrainingResult:
        """
        Executes the training process.
        """
        logger.info("Starting Pacemaker training...")

        try:
            # 1. Prepare Data
            # Assumes data.xyz is already in work_dir or we need to put it there?
            # Config generator expects data path.

            # 2. Generate Config
            if self.config.template_path:
                PacemakerConfigGenerator(self.config.template_path)
            # Render input.yaml

            # 3. Resolve Executable
            executable = self._resolve_executable("pacemaker")

            # 4. Run Training
            # Security: shell=False
            cmd = [executable, "input.yaml"]

            # Validate command construction (basic check since we build it)
            if not isinstance(executable, str): # Should be covered by resolve
                 msg = "Invalid executable path"
                 raise ValueError(msg)

            with (
                open(self.work_dir / "stdout.log", "w") as f_out,
                open(self.work_dir / "stderr.log", "w") as f_err,
            ):
                subprocess.run(
                    cmd, cwd=self.work_dir, stdout=f_out, stderr=f_err, check=True, shell=False
                )

            # 5. Check Output
            potential_file = self.work_dir / "output_potential.yace"  # Hypothetical
            if potential_file.exists():
                return TrainingResult(True, potential_file)
            return TrainingResult(False)

        except subprocess.CalledProcessError:
            logger.exception("Pacemaker process failed")
            return TrainingResult(False)
        except Exception:
            logger.exception("Training failed")
            return TrainingResult(False)

    def _resolve_executable(self, name: str) -> str:
        # Security: Validate name doesn't contain path separators if we expect it in PATH
        if "/" in name or "\\" in name:
             msg = f"Executable name '{name}' must be a simple filename, not a path."
             raise ValueError(msg)

        exe = shutil.which(name)
        if not exe:
            msg = f"Executable {name} not found"
            raise FileNotFoundError(msg)
        return exe

    def select_active_set(self, candidates_path: Path, current_potential: Path) -> list[int]:
        """
        Runs pace_activeset to select structures.
        """
        exe = self._resolve_executable("pace_activeset")

        # Validate paths
        if not candidates_path.exists():
            msg = f"Candidates file {candidates_path} not found"
            raise FileNotFoundError(msg)

        cmd = [exe, "-d", str(candidates_path), "-p", str(current_potential)]

        try:
            subprocess.run(
                cmd, cwd=self.work_dir, capture_output=True, text=True, check=True, shell=False
            )

            # Parse indices from stdout (Mock logic here)
            # In reality, parse result.stdout
            return []

        except subprocess.CalledProcessError:
            logger.exception("Active set selection failed")
            return []
