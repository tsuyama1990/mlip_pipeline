from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase.build import molecule

from mlip_autopipec.config.schemas.generator import (
    DefectConfig,
    DistortionConfig,
    GeneratorConfig,
    NMSConfig,
    SQSConfig,
)
from mlip_autopipec.exceptions import GeneratorError
from mlip_autopipec.generator.molecule import MoleculeGenerator


@pytest.fixture
def base_config():
    return GeneratorConfig(
        sqs=SQSConfig(enabled=True),
        distortion=DistortionConfig(enabled=True),
        nms=NMSConfig(enabled=True),
        defects=DefectConfig(enabled=False)
    )

@pytest.fixture
def molecule_generator(base_config):
    return MoleculeGenerator(base_config)

def test_normal_mode_sampling(molecule_generator):
    # H2O molecule
    h2o = molecule('H2O')

    with patch('ase.vibrations.Vibrations') as MockVib:
        instance = MockVib.return_value
        instance.get_energies.return_value = [0.0]*6 + [0.1, 0.2, 0.3]

        modes = [np.zeros((3,3)) for _ in range(9)]
        modes[6] = np.random.rand(3,3)
        modes[7] = np.random.rand(3,3)
        modes[8] = np.random.rand(3,3)

        instance.get_mode.side_effect = lambda i: modes[i]

        h2o.calc = MagicMock()

        distorted_list = molecule_generator.normal_mode_sampling(h2o, temperature=300, n_samples=5)

        assert len(distorted_list) == 5
        for atoms in distorted_list:
            assert atoms.info['config_type'] == 'nms'
            assert atoms.info['temperature'] == 300
            assert atoms.get_chemical_symbols() == h2o.get_chemical_symbols()

def test_nms_disabled(base_config):
    base_config.nms.enabled = False
    gen = MoleculeGenerator(base_config)
    h2o = molecule('H2O')

    res = gen.normal_mode_sampling(h2o, 300)
    assert res == []

def test_nms_no_calculator(base_config):
    gen = MoleculeGenerator(base_config)
    h2o = molecule('H2O')
    h2o.calc = None

    # We simulate failure to import/instantiate EMT
    with (
        patch('ase.calculators.emt.EMT', side_effect=ImportError("Mocked EMT missing")),
        pytest.raises(GeneratorError, match="NMS requires a calculator")
    ):
        gen.normal_mode_sampling(h2o, 300)
