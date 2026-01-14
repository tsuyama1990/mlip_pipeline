import subprocess
from pathlib import Path

from ase.db import connect
from ase.io import write

from mlip_autopipec.schemas.user_config import TrainerConfig
from mlip_autopipec.utils.pacemaker_utils import generate_pacemaker_input


class PacemakerTrainer:
    """
    A class to train a Pacemaker potential.
    """

    def __init__(self, config: TrainerConfig) -> None:
        self.config = config

    def train_potential(self, database_path: str, output_dir: Path) -> None:
        """
        Trains a Pacemaker potential.

        Args:
            database_path: The path to the ASE database.
            output_dir: The directory to save the trained potential and logs.
        """
        output_dir.mkdir(exist_ok=True)

        # 1. Prepare data
        db = connect(database_path)  # type: ignore[no-untyped-call]
        structures = [row.toatoms() for row in db.select()]
        extxyz_path = output_dir / "structures.extxyz"
        write(str(extxyz_path), structures)

        # 2. Generate pacemaker.in
        pacemaker_input = generate_pacemaker_input(self.config, str(extxyz_path))
        pacemaker_input_path = output_dir / "pacemaker.in"
        pacemaker_input_path.write_text(pacemaker_input)

        # 3. Run training
        subprocess.run(
            ["pacemaker", "--train", str(pacemaker_input_path)],
            check=True,
            cwd=output_dir,
        )
