import pytest
from unittest.mock import patch

from mlip_autopipec.physics.dynamics.lammps import LammpsRunner
from mlip_autopipec.domain_models.config import LammpsConfig, MDParams, PotentialConfig
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.job import JobStatus


@pytest.fixture
def lammps_config():
    return LammpsConfig(command="lmp_mock", cores=1)


@pytest.fixture
def md_params():
    return MDParams(temperature=300, n_steps=100)

@pytest.fixture
def potential_config():
    return PotentialConfig(
        elements=["Si"],
        cutoff=5.0,
        pair_style="hybrid/overlay ace lj/cut 2.5",
        pair_coeff=["* * ace potential.yace Si"]
    )

@pytest.fixture
def silicon_structure(sample_ase_atoms):
    # Overwrite sample with Si if needed, or just use sample
    return Structure.from_ase(sample_ase_atoms)


def test_write_inputs(lammps_config, silicon_structure, md_params, potential_config, tmp_path):
    runner = LammpsRunner(lammps_config, base_work_dir=tmp_path)
    work_dir = tmp_path / "job_01"
    work_dir.mkdir()

    runner._write_inputs(work_dir, silicon_structure, md_params, potential_config)

    assert (work_dir / "data.lammps").exists()
    assert (work_dir / "in.lammps").exists()

    in_file_content = (work_dir / "in.lammps").read_text()
    assert "units" in in_file_content
    assert "metal" in in_file_content
    assert str(md_params.temperature) in in_file_content
    assert str(md_params.n_steps) in in_file_content
    # Check for potential
    assert "pair_style      hybrid/overlay ace lj/cut 2.5" in in_file_content
    assert "pair_coeff      * * ace potential.yace Si" in in_file_content


@patch("mlip_autopipec.infrastructure.io.run_subprocess")
@patch("ase.io.read")
def test_run_success(mock_read, mock_run, lammps_config, silicon_structure, md_params, potential_config, tmp_path):
    # Mock reading the output trajectory
    mock_atoms = silicon_structure.to_ase()
    mock_atoms.positions += 0.1  # Shift positions to simulate movement
    mock_read.return_value = mock_atoms # When index=-1 is used

    # We need to simulate the file creation that LAMMPS would do
    def mock_subprocess_execution(cmd, cwd, timeout, env):
        # Create the dump file in the cwd
        dump_file = cwd / "dump.lammpstrj"
        dump_file.touch()
        return 0, "LAMMPS Done", ""

    mock_run.side_effect = mock_subprocess_execution

    runner = LammpsRunner(lammps_config, base_work_dir=tmp_path)

    result = runner.run(silicon_structure, md_params, potential_config)

    assert result.status == JobStatus.COMPLETED
    assert result.final_structure is not None
    assert result.duration_seconds >= 0
    assert result.work_dir.exists()
    assert result.work_dir.parent == tmp_path


@patch("mlip_autopipec.infrastructure.io.run_subprocess")
def test_run_failed(mock_run, lammps_config, silicon_structure, md_params, potential_config, tmp_path):
    mock_run.return_value = (1, "Error", "Segfault")

    runner = LammpsRunner(lammps_config, base_work_dir=tmp_path)
    result = runner.run(silicon_structure, md_params, potential_config)

    assert result.status == JobStatus.FAILED
    assert "Segfault" in result.log_content
