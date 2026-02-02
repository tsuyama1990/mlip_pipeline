import logging
import os
import subprocess
from pathlib import Path

from mlip_autopipec.config import TrainingConfig

logger = logging.getLogger(__name__)


class PacemakerTrainer:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def train(
        self,
        dataset: Path,
        previous_potential: Path | None,
        output_dir: Path
    ) -> Path:
        """
        Runs pace_train via subprocess or mock if environment variable PYACEMAKER_MOCK_MODE is set.
        Returns the path to the trained potential (standardized to 'potential.yace' in output_dir).
        """
        mock_mode = os.environ.get("PYACEMAKER_MOCK_MODE", "0") == "1"

        output_dir.mkdir(parents=True, exist_ok=True)
        target_output_file = output_dir / "potential.yace"

        if mock_mode:
            logger.info("Running PacemakerTrainer in MOCK MODE")
            target_output_file.touch()
            return target_output_file

        cmd = [self.config.command, "--dataset", str(dataset)]
        if previous_potential:
            cmd.extend(["--initial-potential", str(previous_potential)])

        cmd.extend(["--max-epochs", str(self.config.max_epochs)])
        cmd.extend(["--output-dir", str(output_dir)])

        logger.info(f"Executing: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603
        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed: {e.stderr}")  # noqa: TRY400
            raise

        # Locate the output file
        # Pacemaker might output 'output_potential.yace' or just 'potential.yace' depending on version/config
        candidates = ["output_potential.yace", "potential.yace"]
        found = None
        for name in candidates:
            p = output_dir / name
            if p.exists():
                found = p
                break

        if found:
            # Standardize output name
            if found != target_output_file:
                found.replace(target_output_file)
            return target_output_file

        msg = f"Expected output file not found in {output_dir}. Checked: {candidates}"
        logger.error(msg)
        raise FileNotFoundError(msg)
