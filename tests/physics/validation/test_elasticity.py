from unittest.mock import MagicMock, patch
from pathlib import Path
import numpy as np
from mlip_autopipec.domain_models.config import ValidationConfig, PotentialConfig

def test_elasticity_validator_pass():
    with patch("mlip_autopipec.physics.validation.elasticity.ElasticityValidator._calculate_cij") as MockCij, \
         patch("mlip_autopipec.physics.validation.elasticity.get_reference_structure") as MockGetRef, \
         patch("mlip_autopipec.physics.validation.elasticity.get_calculator"):

        from mlip_autopipec.physics.validation.elasticity import ElasticityValidator

        # Mock Reference Structure
        mock_struct = MagicMock()
        mock_atoms = MagicMock()
        mock_struct.to_ase.return_value = mock_atoms
        MockGetRef.return_value = mock_struct

        # Stable Cubic
        C = np.zeros((6,6))
        C[0,0] = C[1,1] = C[2,2] = 100.0
        C[0,1] = C[0,2] = C[1,0] = C[1,2] = C[2,0] = C[2,1] = 50.0
        C[3,3] = C[4,4] = C[5,5] = 30.0

        MockCij.return_value = C

        pot_config = PotentialConfig(elements=["Si"], cutoff=5.0)
        validator = ElasticityValidator(ValidationConfig(), pot_config)
        result = validator.validate(Path("pot.yace"))

        assert result.overall_status == "PASS"

def test_elasticity_validator_fail():
    with patch("mlip_autopipec.physics.validation.elasticity.ElasticityValidator._calculate_cij") as MockCij, \
         patch("mlip_autopipec.physics.validation.elasticity.get_reference_structure") as MockGetRef, \
         patch("mlip_autopipec.physics.validation.elasticity.get_calculator"):

        from mlip_autopipec.physics.validation.elasticity import ElasticityValidator

        mock_struct = MagicMock()
        mock_atoms = MagicMock()
        mock_struct.to_ase.return_value = mock_atoms
        MockGetRef.return_value = mock_struct

        # Unstable: C11 - C12 < 0
        C = np.zeros((6,6))
        C[0,0] = C[1,1] = C[2,2] = 50.0
        C[0,1] = C[0,2] = C[1,0] = C[1,2] = C[2,0] = C[2,1] = 100.0
        C[3,3] = C[4,4] = C[5,5] = 30.0

        MockCij.return_value = C

        pot_config = PotentialConfig(elements=["Si"], cutoff=5.0)
        validator = ElasticityValidator(ValidationConfig(), pot_config)
        result = validator.validate(Path("pot.yace"))

        assert result.overall_status == "FAIL"
