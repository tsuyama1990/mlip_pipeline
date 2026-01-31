import pytest
import os
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner
from mlip_autopipec.app import app

runner = CliRunner()

@pytest.fixture
def mock_calc():
    calc = MagicMock()
    # Mock PE for EOS
    def get_pe(atoms=None):
        return 0.0 # Will result in flat EOS -> fit failure or 0 bulk modulus?
                   # If energies are all 0, B=0.
    calc.get_potential_energy.side_effect = get_pe

    # Mock Stress for Elasticity
    def get_stress(atoms=None):
        return np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]) # Dummy stress
    calc.get_stress.side_effect = get_stress

    # Mock Forces for Phonon
    def get_forces(atoms=None):
        if atoms is None:
             atoms = calc.atoms
        return np.zeros((len(atoms), 3))
    calc.get_forces.side_effect = get_forces

    return calc

def test_uat_cycle05_validate_flow(tmp_path, mock_calc):
    # Change CWD to tmp_path to isolate files
    current_dir = os.getcwd()
    os.chdir(tmp_path)
    try:
        # 1. Init Project
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        assert Path("config.yaml").exists()

        # 2. Create Dummy Potential
        pot_path = Path("dummy.yace")
        pot_path.touch()

        # 3. Mock External dependencies
        # We patch the `get_lammps_calculator` in each validator module because they import it
        with patch("mlip_autopipec.physics.validation.eos.get_lammps_calculator", return_value=mock_calc), \
             patch("mlip_autopipec.physics.validation.elasticity.get_lammps_calculator", return_value=mock_calc), \
             patch("mlip_autopipec.physics.validation.phonon.get_lammps_calculator", return_value=mock_calc), \
             patch("mlip_autopipec.physics.validation.phonon.Phonopy") as MockPhonopy:

            # Setup Phonopy mock
            mock_phonopy_inst = MockPhonopy.return_value
            # return positive frequencies
            mock_phonopy_inst.get_mesh_dict.return_value = {'frequencies': np.array([[1.0, 2.0]])}
            # mock plotting
            mock_fig = MagicMock()
            mock_phonopy_inst.plot_band_structure.return_value = mock_fig

            # 4. Run Validate
            result = runner.invoke(app, ["validate", "--potential", "dummy.yace"])

            print(result.stdout)
            assert result.exit_code == 0

            # Check Output
            assert "Validation Finished" in result.stdout
            assert "Metrics:" in result.stdout
            assert "Report generated at: validation_report.html" in result.stdout

            # Check Report File
            assert Path("validation_report.html").exists()
            content = Path("validation_report.html").read_text()
            assert "Validation Report: dummy.yace" in content

    finally:
        os.chdir(current_dir)
