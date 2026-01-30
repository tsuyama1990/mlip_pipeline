import pytest
from mlip_autopipec.domain_models.calculation import DFTConfig, SCFError
from mlip_autopipec.physics.dft.recovery import RecoveryHandler

def test_recovery_scf_mixing():
    handler = RecoveryHandler()
    original_config = DFTConfig(
        command="pw.x",
        pseudopotentials={},
        ecutwfc=30.0,
        mixing_beta=0.7
    )

    error = SCFError("convergence not achieved")

    # First attempt to recover should reduce mixing beta
    new_config = handler.apply_fix(original_config, error, attempt=1)
    assert new_config.mixing_beta < 0.7
    assert new_config.mixing_beta == pytest.approx(0.7 * 0.7) # Assuming 0.7 reduction factor

def test_recovery_scf_smearing():
    handler = RecoveryHandler()
    original_config = DFTConfig(
        command="pw.x",
        pseudopotentials={},
        ecutwfc=30.0,
        mixing_beta=0.1 # already low
    )

    error = SCFError("convergence not achieved")

    # If mixing beta is low, maybe it increases temperature/smearing
    # This depends on implementation strategy, but let's assume we have a strategy
    # that switches strategy after some attempts

    # Let's say attempt 2 tries changing diagonalization
    new_config = handler.apply_fix(original_config, error, attempt=2)
    # Just verify it returns a valid config and *something* changed
    assert isinstance(new_config, DFTConfig)
    assert new_config != original_config

def test_recovery_give_up():
    handler = RecoveryHandler(max_retries=3)
    config = DFTConfig(command="pw.x", pseudopotentials={}, ecutwfc=30.0)
    error = SCFError("fatal")

    with pytest.raises(RuntimeError, match="Max retries.*reached"):
        handler.apply_fix(config, error, attempt=4)
