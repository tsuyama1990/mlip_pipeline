import pytest
from mlip_autopipec.domain_models.calculation import DFTConfig, SCFError, MemoryError
from mlip_autopipec.physics.dft.recovery import RecoveryHandler


@pytest.fixture
def base_config():
    return DFTConfig(
        command="pw.x",
        pseudopotentials={},
        ecutwfc=30.0,
        kspacing=0.04,
        mixing_beta=0.7,
        smearing="gaussian",
        degauss=0.02,
    )


def test_recovery_scf_mix_beta(base_config):
    handler = RecoveryHandler()

    # First failure -> Reduce mixing beta
    new_config = handler.apply_fix(
        base_config, SCFError("convergence not achieved"), attempt=1
    )

    # Assert mixing beta is reduced (e.g., 0.7 -> 0.3)
    assert new_config.mixing_beta < base_config.mixing_beta
    assert new_config.mixing_beta == 0.3


def test_recovery_scf_smearing(base_config):
    handler = RecoveryHandler()

    # Second failure (or specific logic) -> Increase temperature/smearing
    # Let's assume attempt 2 tries smearing change
    new_config = handler.apply_fix(
        base_config, SCFError("convergence not achieved"), attempt=2
    )

    assert new_config.smearing == "mv" or new_config.degauss > base_config.degauss


def test_recovery_no_fix(base_config):
    handler = RecoveryHandler()

    with pytest.raises(MemoryError):
        # We might not have a fix for MemoryError yet, so it should re-raise
        handler.apply_fix(base_config, MemoryError("OOM"), attempt=1)

def test_recovery_scf_mix_beta_aggressive(base_config):
    handler = RecoveryHandler()
    # Attempt 3 -> mixing_beta = 0.1
    new_config = handler.apply_fix(base_config, SCFError("convergence not achieved"), attempt=3)
    assert new_config.mixing_beta == 0.1
