import pytest
from mlip_autopipec.domain_models.calculation import (
    DFTConfig,
    MemoryError,
    RecoveryConfig,
    SCFError,
)
from mlip_autopipec.physics.dft.recovery import RecoveryHandler


@pytest.fixture
def recovery_config():
    # Define custom strategies
    return RecoveryConfig(
        scf_strategies=[
            {"mixing_beta": 0.5},
            {"mixing_beta": 0.2},
            {"smearing": "mv", "degauss": 0.05},
        ],
        memory_strategies=[
            {"diagonalization": "cg"},
        ],
    )


def test_recovery_scf_strategy_configurable(recovery_config):
    handler = RecoveryHandler(recovery_config)

    # Attempt 1: Custom mixing beta 0.5
    params = {}
    new_params = handler.apply_fix(params, SCFError("Convergence failed"), attempt=1)
    assert new_params["mixing_beta"] == 0.5

    # Attempt 2: Custom mixing beta 0.2
    new_params = handler.apply_fix(params, SCFError("Convergence failed"), attempt=2)
    assert new_params["mixing_beta"] == 0.2

    # Attempt 3: Custom smearing
    new_params = handler.apply_fix(params, SCFError("Convergence failed"), attempt=3)
    assert new_params["smearing"] == "mv"
    assert new_params["degauss"] == 0.05


def test_recovery_memory_strategy_configurable(recovery_config):
    handler = RecoveryHandler(recovery_config)

    new_params = handler.apply_fix({}, MemoryError("OOM"), attempt=1)
    assert new_params.get("diagonalization") == "cg"


def test_recovery_unknown_error(recovery_config):
    handler = RecoveryHandler(recovery_config)

    with pytest.raises(ValueError):
        handler.apply_fix({}, Exception("Unknown"), attempt=1)


def test_max_retries_configurable(recovery_config):
    handler = RecoveryHandler(recovery_config)

    # Memory strategies has only 1 item. Attempt 2 should fail.
    with pytest.raises(ValueError, match="No more recovery strategies"):
        handler.apply_fix({}, MemoryError("OOM"), attempt=2)
