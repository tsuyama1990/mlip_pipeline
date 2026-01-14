
# Copyright (C) 2024-present by the LICENSE file authors.
#
# This file is part of MLIP-AutoPipe.
#
# MLIP-AutoPipe is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MLIP-AutoPipe is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MLIP-AutoPipe.  If not, see <https://www.gnu.org/licenses/>.
"""This module provides the PacemakerTrainer class for training MLIPs."""
import subprocess
from pathlib import Path

from ase.db import connect
from ase.io import write

from mlip_autopipec.schemas.user_config import TrainerConfig
from mlip_autopipec.utils.pacemaker_utils import generate_pacemaker_input

from mlip_autopipec.utils.logging import get_logger

logger = get_logger(__name__)


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
        logger.info("Starting Pacemaker training...")
        result = subprocess.run(
            ["pacemaker", "--train", str(pacemaker_input_path)],
            check=True,
            cwd=output_dir,
            capture_output=True,
            text=True,
        )
        logger.info("Pacemaker training finished.")
        logger.debug("Pacemaker stdout:\n%s", result.stdout)
        if result.stderr:
            logger.warning("Pacemaker stderr:\n%s", result.stderr)
