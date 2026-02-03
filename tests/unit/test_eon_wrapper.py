from pathlib import Path
from unittest.mock import patch

import pytest

from mlip_autopipec.config.config_model import EonConfig
from mlip_autopipec.physics.dynamics.eon_wrapper import EonWrapper


@pytest.fixture
def eon_config() -> EonConfig:
    return EonConfig(
        command="mock_eon_client",
        parameters={"temperature": 300, "process_search_min_barrier": 0.1}
    )


def test_eon_wrapper_init(eon_config: EonConfig) -> None:
    wrapper = EonWrapper(eon_config)
    assert wrapper.config == eon_config


def test_run_akmc_setup(eon_config: EonConfig, tmp_path: Path) -> None:
    wrapper = EonWrapper(eon_config)

    potential_path = tmp_path / "potential.yace"
    potential_path.touch()
    structure_path = tmp_path / "structure.xyz"
    structure_path.write_text("2\n\nCu 0 0 0\nCu 2 0 0")

    work_dir = tmp_path / "work"
    work_dir.mkdir()

    with patch("subprocess.run") as mock_run:
        # Mock successful run
        mock_run.return_value.returncode = 0

        wrapper.run_akmc(potential_path, structure_path, work_dir)

        # Check if config.ini was created
        assert (work_dir / "config.ini").exists()
        # Check if driver was copied (not checking content, just existence logic)
        assert (work_dir / "pace_driver.py").exists()

        # Check if eon client was called
        mock_run.assert_called_once()
