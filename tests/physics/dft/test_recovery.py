from pathlib import Path
import pytest
from mlip_autopipec.domain_models.calculation import DFTConfig, SCFError, DFTError, MemoryError
from mlip_autopipec.physics.dft.recovery import RecoveryHandler

@pytest.fixture
def base_config():
    return DFTConfig(
        command="pw.x",
        pseudopotentials={"Si": Path("Si.UPF")},
        ecutwfc=30.0,
        kspacing=0.04,
        mixing_beta=0.7,
        smearing="mv"
    )

def test_recovery_scf_failure(base_config):
    handler = RecoveryHandler()
    error = SCFError("Convergence not achieved")

    # First attempt should reduce mixing beta
    new_config = handler.apply_fix(base_config, error, attempt=1)
    assert new_config.mixing_beta < base_config.mixing_beta
    assert new_config.mixing_beta == 0.3 # As per SPEC example

def test_recovery_max_attempts(base_config):
    handler = RecoveryHandler()
    error = SCFError("Convergence not achieved")

    with pytest.raises(DFTError, match="Max recovery attempts reached"):
        handler.apply_fix(base_config, error, attempt=10)

def test_unknown_error(base_config):
    handler = RecoveryHandler()
    error = DFTError("Some unknown error")

    # Should probably raise the error again if no fix found
    with pytest.raises(DFTError):
        handler.apply_fix(base_config, error, attempt=1)
