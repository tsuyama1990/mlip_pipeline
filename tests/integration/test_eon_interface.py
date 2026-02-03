from pathlib import Path

from mlip_autopipec.config.config_model import EonConfig
from mlip_autopipec.physics.dynamics.eon_wrapper import EonWrapper


def test_eon_integration_flow(tmp_path: Path) -> None:
    # Setup
    config = EonConfig(command="echo Running EON")
    wrapper = EonWrapper(config)

    potential_path = tmp_path / "potential.yace"
    potential_path.touch()

    structure_path = tmp_path / "start.xyz"
    # Create valid XYZ
    structure_path.write_text("2\n\nCu 0 0 0\nCu 2 0 0")

    work_dir = tmp_path / "akmc_run"
    # run_akmc creates dir if not exists

    # Execution
    candidates = wrapper.run_akmc(potential_path, structure_path, work_dir)

    # Assertions
    assert isinstance(candidates, list)
    assert len(candidates) == 0  # echo doesn't produce output

    # Check artifacts
    assert (work_dir / "config.ini").exists()
    assert (work_dir / "pace_driver.py").exists()
    assert (work_dir / "potential.yace").exists()
    assert (work_dir / "pos.con").exists()
