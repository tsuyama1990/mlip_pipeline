import pytest
import numpy as np
from ase import Atoms
from unittest.mock import MagicMock
from pathlib import Path

from mlip_autopipec.physics.validation.elasticity import ElasticityValidator
from mlip_autopipec.domain_models.config import ValidationConfig, PotentialConfig
from mlip_autopipec.domain_models.structure import Structure

def test_elasticity_validator_success():
    val_config = ValidationConfig()
    pot_config = PotentialConfig(elements=["Al"], cutoff=5.0)
    validator = ElasticityValidator(val_config, pot_config, potential_path=Path("dummy.yace"))

    # Mock internal calculation of C_ij
    # Born stability for Cubic:
    # C11 - C12 > 0
    # C11 + 2C12 > 0
    # C44 > 0
    C_ij = np.zeros((6,6))
    C_ij[0,0] = C_ij[1,1] = C_ij[2,2] = 100.0 # C11
    C_ij[0,1] = C_ij[1,0] = C_ij[0,2] = C_ij[2,0] = C_ij[1,2] = C_ij[2,1] = 50.0 # C12
    C_ij[3,3] = C_ij[4,4] = C_ij[5,5] = 40.0 # C44

    validator._calculate_stiffness = MagicMock(return_value=C_ij)

    structure = Structure.from_ase(Atoms("Al", cell=[4,4,4], pbc=True))

    metric = validator.validate(structure)

    assert metric.passed is True
    assert metric.message is None or "Stable" in metric.message

def test_elasticity_validator_failure():
    val_config = ValidationConfig()
    pot_config = PotentialConfig(elements=["Al"], cutoff=5.0)
    validator = ElasticityValidator(val_config, pot_config, potential_path=Path("dummy.yace"))

    # Unstable C_ij (C44 < 0)
    C_ij = np.zeros((6,6))
    C_ij[0,0] = 100.0
    C_ij[3,3] = -10.0

    validator._calculate_stiffness = MagicMock(return_value=C_ij)

    structure = Structure.from_ase(Atoms("Al", cell=[4,4,4], pbc=True))

    metric = validator.validate(structure)

    assert metric.passed is False
