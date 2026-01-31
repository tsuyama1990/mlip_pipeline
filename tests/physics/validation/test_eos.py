import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from mlip_autopipec.domain_models.config import ValidationConfig, PotentialConfig

def test_eos_validator_pass():
    with patch("mlip_autopipec.physics.validation.eos.EquationOfState") as MockEOS, \
         patch("mlip_autopipec.physics.validation.eos.bulk") as MockBulk, \
         patch("mlip_autopipec.physics.validation.eos.get_calculator"):

        from mlip_autopipec.physics.validation.eos import EOSValidator

        # Mock EOS fit
        mock_eos_instance = MockEOS.return_value
        # v0, e0, B, dB/dP (B in eV/A^3)
        # 100 GPa ~ 0.62 eV/A^3
        mock_eos_instance.fit.return_value = (10.0, -5.0, 0.624, 4.0)

        # Mock Atoms
        mock_atoms = MagicMock()
        mock_atoms.get_potential_energy.return_value = -5.0
        mock_atoms.get_volume.return_value = 10.0
        mock_atoms.copy.return_value = mock_atoms # Needed for at.copy()
        MockBulk.return_value = mock_atoms

        pot_config = PotentialConfig(elements=["Si"], cutoff=5.0)
        validator = EOSValidator(ValidationConfig(), pot_config)
        result = validator.validate(Path("pot.yace"))

        assert result.overall_status == "PASS"
        b_metric = next(m for m in result.metrics if m.name == "Bulk Modulus")
        assert b_metric.value == pytest.approx(100.0, rel=0.1)

def test_eos_validator_fail():
    with patch("mlip_autopipec.physics.validation.eos.EquationOfState") as MockEOS, \
         patch("mlip_autopipec.physics.validation.eos.bulk") as MockBulk, \
         patch("mlip_autopipec.physics.validation.eos.get_calculator"):

        from mlip_autopipec.physics.validation.eos import EOSValidator

        # B = -0.1 (Unstable)
        mock_eos_instance = MockEOS.return_value
        mock_eos_instance.fit.return_value = (10.0, -5.0, -0.1, 4.0)

        mock_atoms = MagicMock()
        mock_atoms.copy.return_value = mock_atoms
        MockBulk.return_value = mock_atoms

        pot_config = PotentialConfig(elements=["Si"], cutoff=5.0)
        validator = EOSValidator(ValidationConfig(), pot_config)
        result = validator.validate(Path("pot.yace"))

        assert result.overall_status == "FAIL"
