import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from mlip_autopipec.config.schemas.inference import InferenceConfig
from ase import Atoms
import numpy as np

# UAT-06-01: LAMMPS Simulation Launch
def test_uat_06_01_launch_simulation(tmp_path):
    # GIVEN an InferenceConfig with valid potential path
    pot_path = tmp_path / "model.yace"
    pot_path.touch()

    config = InferenceConfig(
        lammps_executable="lmp_serial",
        potential_path=pot_path,
        temperature=1000,
        steps=100
    )

    # WHEN we launch the simulation (Mocked)
    with patch("mlip_autopipec.inference.lammps_runner.LammpsRunner._execute_lammps") as mock_exec:
        mock_exec.return_value = None

        from mlip_autopipec.inference.lammps_runner import LammpsRunner
        runner = LammpsRunner(config)
        # Mock atoms
        atoms = Atoms("Al", cell=[10,10,10])

        # Mock UQ to return nothing
        with patch("mlip_autopipec.inference.lammps_runner.UncertaintyChecker") as MockUQ:
            MockUQ.return_value.parse_dump.return_value = []

            result = runner.run(atoms)

            # THEN it should succeed
            assert result.succeeded
            # AND the script should have been generated (implicitly checked by _execute_lammps being called)
            assert mock_exec.called

# UAT-06-02: Uncertainty Mining
def test_uat_06_02_uncertainty_mining(tmp_path):
    # GIVEN a simulation where uncertainty is found
    # Need to config with valid potential even if mocked execute, because run() checks existence
    pot_path = tmp_path / "model.yace"
    pot_path.touch()

    config = InferenceConfig(temperature=2000, potential_path=pot_path)

    with patch("mlip_autopipec.inference.lammps_runner.LammpsRunner._execute_lammps"):
         with patch("mlip_autopipec.inference.lammps_runner.UncertaintyChecker") as MockUQ:
            # Setup UQ to return a structure
            bad_atom = Atoms("Al", cell=[10,10,10])
            bad_atom.info['src_md_step'] = 500
            # Mock gamma as numpy array
            bad_atom.arrays['gamma'] = np.array([6.0])

            MockUQ.return_value.parse_dump.return_value = [bad_atom]
            MockUQ.return_value.max_gamma = 6.0

            from mlip_autopipec.inference.lammps_runner import LammpsRunner
            runner = LammpsRunner(config)
            result = runner.run(Atoms("Al", cell=[10,10,10]))

            # THEN we should get uncertain structures
            assert len(result.uncertain_structures) > 0
            # AND max gamma should be reported
            assert result.max_gamma_observed >= 5.0

# UAT-06-03: Trajectory Analysis
def test_uat_06_03_trajectory_analysis(tmp_path):
    # GIVEN a log file
    log_file = tmp_path / "log.lammps"
    log_file.write_text("Step Temp\n0 300\n100 310")

    # WHEN we analyze it
    from mlip_autopipec.inference.analysis import AnalysisUtils
    utils = AnalysisUtils(log_file)
    props = utils.get_properties()

    # THEN we get stats
    assert "temperature" in props
    assert props["temperature"] == 305.0

# UAT-06-04: Ensemble Control
def test_uat_06_04_ensemble_control(tmp_path):
    # GIVEN NPT config
    pot_path = tmp_path / "model.yace"
    pot_path.touch()
    config = InferenceConfig(
        temperature=300,
        ensemble="npt",
        pressure=100.0,
        potential_path=pot_path
    )
    from mlip_autopipec.inference.inputs import ScriptGenerator
    gen = ScriptGenerator(config)
    script = gen.generate(Atoms("Al", cell=[10,10,10]), Path("."), Path("s.data"))

    # THEN fix npt is used
    assert "fix             1 all npt" in script
