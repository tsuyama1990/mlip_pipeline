import numpy as np
import pytest
from ase.build import molecule
from unittest.mock import MagicMock, patch
import sys

from mlip_autopipec.config.schemas.generator import GeneratorConfig
from mlip_autopipec.generator.molecule import MoleculeGenerator


@pytest.fixture
def molecule_generator():
    config = GeneratorConfig()
    return MoleculeGenerator(config)

def test_normal_mode_sampling(molecule_generator):
    # H2O molecule
    h2o = molecule('H2O')

    # We patch the class used in the module: 'mlip_autopipec.generator.molecule.Vibrations'
    # Wait, the import inside the method is:
    # from ase.vibrations import Vibrations
    # So if we want to patch it, we must patch where it's looked up.
    # Since it is a local import inside the method, 'patch' on module level might not work if not imported at top level?
    # Actually, patch works on where the name is resolved.
    # If I patch 'ase.vibrations.Vibrations', it should work regardless of where it is imported?

    with patch('ase.vibrations.Vibrations') as MockVib:
        instance = MockVib.return_value
        # Setup energies (real)
        # 3N = 9 dof. 6 zero modes (trans+rot), 3 vib modes.
        instance.get_energies.return_value = [0.0]*6 + [0.1, 0.2, 0.3]

        # Setup modes
        # get_mode(i) returns ndarray (N, 3)
        modes = [np.zeros((3,3)) for _ in range(9)]
        modes[6] = np.random.rand(3,3)
        modes[7] = np.random.rand(3,3)
        modes[8] = np.random.rand(3,3)

        instance.get_mode.side_effect = lambda i: modes[i]

        h2o.calc = MagicMock()

        # Generate samples
        # We ask for 300K
        distorted_list = molecule_generator.normal_mode_sampling(h2o, temperature=300, n_samples=5)

        assert len(distorted_list) == 5
        for atoms in distorted_list:
            assert atoms.info['config_type'] == 'nms'
            assert atoms.info['temperature'] == 300
            # Symbols should match
            assert atoms.get_chemical_symbols() == h2o.get_chemical_symbols()
