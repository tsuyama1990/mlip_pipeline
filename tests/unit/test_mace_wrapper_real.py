import pytest
from unittest.mock import MagicMock, patch
import sys
from ase import Atoms
import numpy as np
from mlip_autopipec.surrogate.mace_wrapper import MaceWrapper

@pytest.fixture
def mock_mace_calc():
    with patch.dict(sys.modules, {'mace.calculators': MagicMock()}):
        mock_calc = MagicMock()
        sys.modules['mace.calculators'].MACECalculator = MagicMock(return_value=mock_calc)
        yield sys.modules['mace.calculators'].MACECalculator

def test_load_model_mace(mock_mace_calc):
    wrapper = MaceWrapper(model_type="mace_mp")
    wrapper.load_model("medium", "cpu")
    mock_mace_calc.assert_called_with(model_paths="medium", device="cpu", default_dtype="float32")
    assert wrapper.model is not None

def test_load_model_mace_fail():
    with patch.dict(sys.modules, {'mace.calculators': None}): # Simulate missing module by removing it or mock raising ImportError
        with patch('builtins.__import__', side_effect=ImportError("No module named mace")):
             # This is hard to mock correctly for existing imports.
             pass

    # Easier: mock import inside the method if I moved import there (I did).
    with patch.dict(sys.modules, {'mace.calculators': MagicMock()}):
        sys.modules['mace.calculators'].MACECalculator.side_effect = Exception("Download failed")
        wrapper = MaceWrapper(model_type="mace_mp")
        with pytest.raises(RuntimeError, match="Failed to load MACE model"):
            wrapper.load_model("medium", "cpu")

def test_compute_energy_forces_mace(mock_mace_calc):
    # Mock the calculator instance
    calc_instance = mock_mace_calc.return_value
    # Mock get_potential_energy and get_forces
    calc_instance.get_potential_energy.return_value = -3.5
    calc_instance.get_forces.return_value = np.zeros((2, 3))

    wrapper = MaceWrapper(model_type="mace_mp")
    wrapper.load_model("medium", "cpu")

    atoms = Atoms('H2', positions=[[0,0,0], [0,0,0.74]])
    # We need to ensure atoms.copy() is mocked or handles calc correctly.
    # The wrapper does: at = atoms.copy(); at.calc = self.model
    # So we need to ensure at.calc works.

    # We patch Atoms.copy to return a mock or just rely on ASE copy which copies everything but calculator?
    # ASE atoms.copy() does not copy calculator.

    # But wrapper assigns `at.calc = self.model`.
    # And calls `at.get_potential_energy()`.
    # This calls `self.model.get_potential_energy(at)`.
    # So our mock calculator needs to handle that.

    # When `at.calc = calc_instance` is set, `at.get_potential_energy()` calls `calc_instance.get_potential_energy(atoms=at)`.

    energies, forces = wrapper.compute_energy_forces([atoms])

    assert len(energies) == 1
    assert energies[0] == -3.5
    assert len(forces) == 1

def test_compute_descriptors_dscribe():
    # Mock dscribe
    with patch.dict(sys.modules, {'dscribe.descriptors': MagicMock()}):
        mock_soap = MagicMock()
        sys.modules['dscribe.descriptors'].SOAP = MagicMock(return_value=mock_soap)
        mock_soap.create.return_value = np.random.rand(2, 10)

        wrapper = MaceWrapper(model_type="mace_mp")
        atoms = [Atoms('H2'), Atoms('H2')]

        desc = wrapper.compute_descriptors(atoms)

        assert desc.shape == (2, 10)
        mock_soap.create.assert_called()
