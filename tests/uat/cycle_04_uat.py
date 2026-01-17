"""
UAT Test for the LammpsRunner module.
"""

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from ase.build import bulk

from mlip_autopipec.config.models import (
    InferenceConfig,
    MDConfig,
    UncertaintyConfig,
)
from mlip_autopipec.modules.inference import LammpsRunner


def setup_mock_lammps_run(mocker: MagicMock, working_dir: Path) -> None:
    """Sets up the mock for a successful LAMMPS run with uncertainty."""

    def side_effect_subprocess_run(*args: Any, **kwargs: Any) -> MagicMock:
        # Create a fake trajectory file
        traj_file = working_dir / "dump.custom"
        with traj_file.open("w") as f:
            f.write("ITEM: TIMESTEP\n10\n")
            f.write("ITEM: NUMBER OF ATOMS\n2\n")
            f.write("ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n")
            f.write("ITEM: ATOMS id type x y z\n1 1 1.0 1.0 1.0\n2 1 2.0 2.0 2.0\n")

        # Create a fake uncertainty file
        uncert_file = working_dir / "uncertainty.dump"
        with uncert_file.open("w") as f:
            f.write("ITEM: TIMESTEP\n10\n")
            f.write("ITEM: NUMBER OF ATOMS\n2\n")
            f.write("ITEM: ATOMS c_uncert[1]\n3.0\n1.0\n")  # High uncertainty
        return MagicMock(returncode=0)

    mocker.patch("subprocess.run", side_effect=side_effect_subprocess_run)


def main(mocker: MagicMock) -> None:
    """
    Main UAT test function.
    """
    # GIVEN a LammpsRunner with a valid configuration
    mock_config = InferenceConfig(
        lammps_executable=Path("/usr/bin/lmp"),
        potential_path=Path("Si.yace"),
        md_params=MDConfig(),
        uncertainty_params=UncertaintyConfig(threshold=2.5),
    )
    runner = LammpsRunner(inference_config=mock_config)

    # WHEN the simulation is run
    with tempfile.TemporaryDirectory() as temp_dir:
        working_dir = Path(temp_dir)
        setup_mock_lammps_run(mocker, working_dir)
        initial_structure = bulk("Si", "diamond", a=5.43)
        result = runner.run(initial_structure)

        # THEN an uncertain structure is returned
        assert result is not None, "An uncertain structure should be detected."
        assert result.metadata["uncertain_timestep"] == 10, "Incorrect timestep for uncertainty."


if __name__ == "__main__":
    with patch("subprocess.run") as mock_run:
        main(MagicMock())
