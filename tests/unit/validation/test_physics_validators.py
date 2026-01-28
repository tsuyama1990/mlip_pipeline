from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms

# Import the module explicitly for patching
import mlip_autopipec.validation.phonon
from mlip_autopipec.config.schemas.validation import EOSConfig, PhononConfig
from mlip_autopipec.validation.eos import EOSValidator
from mlip_autopipec.validation.phonon import PhononValidator


@pytest.fixture
def dummy_atoms():
    return Atoms("Si2", positions=[[0, 0, 0], [1.36, 1.36, 1.36]], cell=[5.43]*3, pbc=True)


@pytest.fixture
def work_dir(tmp_path):
    d = tmp_path / "validation"
    d.mkdir()
    return d


class TestPhononValidator:
    def test_phonon_pass(self, dummy_atoms, work_dir):
        # Manual patching to ensure it works
        original_phonopy = mlip_autopipec.validation.phonon.Phonopy
        mock_phonopy_cls = MagicMock()
        mlip_autopipec.validation.phonon.Phonopy = mock_phonopy_cls

        try:
            # Mock dependencies
            with patch("mlip_autopipec.validation.phonon.load_calculator") as mock_load_calc:
                mock_calc = MagicMock()
                mock_load_calc.return_value = mock_calc

                mock_phonopy_instance = mock_phonopy_cls.return_value

                # Mock produce_force_constants to do nothing (avoid solver error)
                mock_phonopy_instance.produce_force_constants.return_value = None

                # Mock frequencies
                # Note: get_mesh_dict needs to return a dict with frequencies
                mock_phonopy_instance.get_mesh_dict.return_value = {
                    "frequencies": np.array([[1.0, 2.0], [1.5, 2.5]])
                }

                # Mock supercells
                mock_sc = MagicMock()
                mock_sc.symbols = ["Si", "Si"]
                mock_sc.scaled_positions = [[0,0,0], [0.5,0.5,0.5]]
                mock_sc.cell = [[1,0,0],[0,1,0],[0,0,1]]

                mock_phonopy_instance.supercells_with_displacements = [mock_sc]

                config = PhononConfig(enabled=True)
                validator = PhononValidator(config, work_dir)

                result = validator.validate(dummy_atoms, Path("pot.yace"))

                assert result.module == "phonon"
                assert result.passed is True
                assert len(result.metrics) > 0
                assert result.metrics[0].name == "min_frequency"
        finally:
            mlip_autopipec.validation.phonon.Phonopy = original_phonopy

    def test_phonon_fail_imaginary(self, dummy_atoms, work_dir):
        original_phonopy = mlip_autopipec.validation.phonon.Phonopy
        mock_phonopy_cls = MagicMock()
        mlip_autopipec.validation.phonon.Phonopy = mock_phonopy_cls

        try:
            with patch("mlip_autopipec.validation.phonon.load_calculator") as mock_load_calc:
                mock_calc = MagicMock()
                mock_load_calc.return_value = mock_calc

                mock_phonopy_instance = mock_phonopy_cls.return_value

                # Mock produce_force_constants
                mock_phonopy_instance.produce_force_constants.return_value = None

                mock_phonopy_instance.get_mesh_dict.return_value = {
                    "frequencies": np.array([[-1.0, 2.0], [1.5, 2.5]])
                }

                mock_sc = MagicMock()
                mock_sc.symbols = ["Si", "Si"]
                mock_sc.scaled_positions = [[0,0,0], [0.5,0.5,0.5]]
                mock_sc.cell = [[1,0,0],[0,1,0],[0,0,1]]

                mock_phonopy_instance.supercells_with_displacements = [mock_sc]

                config = PhononConfig(enabled=True)
                validator = PhononValidator(config, work_dir)

                result = validator.validate(dummy_atoms, Path("pot.yace"))

                assert result.passed is False
        finally:
             mlip_autopipec.validation.phonon.Phonopy = original_phonopy


class TestElasticityValidator:
    # ... Skipping detailed check for now ...
    pass


class TestEOSValidator:
    @patch("mlip_autopipec.validation.eos.load_calculator")
    @patch("mlip_autopipec.validation.eos.EquationOfState")
    def test_eos_pass(self, mock_eos_cls, mock_load_calc, dummy_atoms, work_dir):
        mock_calc = MagicMock()
        mock_load_calc.return_value = mock_calc

        # Fix mock calculator for ASE
        mock_calc.get_stress.return_value = np.zeros(6)
        mock_calc.get_forces.return_value = np.zeros((len(dummy_atoms), 3))
        mock_calc.get_potential_energy.return_value = -100.0

        # Mock EOS fit
        mock_eos = mock_eos_cls.return_value
        mock_eos.fit.return_value = (100.0, -100.0, 1.0) # v0, e0, B (eV/A^3)
        mock_eos.plot.return_value = None

        config = EOSConfig(enabled=True)
        validator = EOSValidator(config, work_dir)

        result = validator.validate(dummy_atoms, Path("pot.yace"))

        assert result.passed is True
        assert result.metrics[0].name == "bulk_modulus"
        # B = 1.0 eV/A^3 = 160.2 GPa
        assert result.metrics[0].value > 100.0
