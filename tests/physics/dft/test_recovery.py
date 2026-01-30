import pytest
from mlip_autopipec.physics.dft.recovery import RecoveryHandler
from mlip_autopipec.domain_models.calculation import RecoveryConfig, SCFError, DFTConfig

def test_recovery_strategy_selection():
    config = RecoveryConfig(
        strategies={
            "SCFError": [
                {"mixing_beta": 0.3},
                {"smearing": "mv"},
            ]
        }
    )
    dft_config = DFTConfig(
        command=["pw.x"],
        pseudopotentials={},
        recovery=config
    )

    handler = RecoveryHandler(config)

    # First failure
    new_config, attempt = handler.apply_fix(dft_config, SCFError("convergence not achieved"), attempt=0)
    assert new_config.mixing_beta == 0.3
    assert attempt == 1

    # Second failure
    new_config_2, attempt_2 = handler.apply_fix(new_config, SCFError("convergence not achieved"), attempt=1)
    assert new_config_2.smearing == "mv"
    assert attempt_2 == 2

    # Exhausted retries
    with pytest.raises(SCFError):
        handler.apply_fix(new_config_2, SCFError("convergence not achieved"), attempt=2)
