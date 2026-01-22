import numpy as np
from ase import Atoms

from mlip_autopipec.surrogate.mace_wrapper import MaceWrapper


def test_mace_wrapper_mock_energy_forces():
    # Test the mock implementation
    wrapper = MaceWrapper(model_type="mock")
    wrapper.load_model("dummy_path", "cpu")

    atoms_list = [Atoms('H2', positions=[[0,0,0], [0,0,0.75]])]
    energy, forces = wrapper.compute_energy_forces(atoms_list)

    assert isinstance(energy, np.ndarray)
    assert energy.shape == (1,)
    assert isinstance(forces, list)
    assert len(forces) == 1
    assert forces[0].shape == (2, 3)

def test_mace_wrapper_mock_descriptors():
    wrapper = MaceWrapper(model_type="mock")
    atoms_list = [Atoms('H2'), Atoms('O2')]
    descriptors = wrapper.compute_descriptors(atoms_list)

    assert isinstance(descriptors, np.ndarray)
    assert descriptors.shape[0] == 2
    assert descriptors.shape[1] > 0
