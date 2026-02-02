import logging
import os
import subprocess
from pathlib import Path

from mlip_autopipec.config.config_model import TrainingConfig


class PacemakerTrainer:
    """
    Wrapper for the Pacemaker training executable (pace_train).
    """
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)

    def train(self, dataset: Path, previous_potential: Path | None = None) -> Path:
        """
        Runs pace_train via subprocess.
        In mock mode, just touches the output file.
        """
        output_file = Path("output_potential.yace")

        # Mock Mode
        if os.environ.get("PYACEMAKER_MOCK_MODE") == "1":
            self.logger.info("MOCK MODE: Simulating training...")
            output_file.touch()
            return output_file

        cmd = [self.config.command, "--dataset", str(dataset)]
        cmd.extend(["--max-epochs", str(self.config.max_epochs)])

        if previous_potential:
             cmd.extend(["--initial-potential", str(previous_potential)])

        self.logger.info(f"Executing: {' '.join(cmd)}")
        # We use check=True to raise CalledProcessError on failure
        subprocess.run(cmd, check=True)  # noqa: S603

        # Verification
        # Real pace_train might allow configuring output name.
        # For now, we assume it produces output_potential.yace or we rename it?
        # The spec says "return Path('output_potential.yace')".
        # If real execution doesn't produce it, this will fail in Cycle 06/Production testing.
        # But for Cycle 01 we rely on Mock or simple assumptions.

        if not output_file.exists():
             # Try to find what it produced? or Just fail.
             # Standard pace_train behavior needs to be confirmed.
             # Usually output.yace.
             # Let's assume output_potential.yace for now as per spec.
             msg = f"Training failed to produce {output_file}"
             raise FileNotFoundError(msg)

        return output_file
