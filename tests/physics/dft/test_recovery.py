import pytest
from mlip_autopipec.domain_models.calculation import SCFError, MemoryError, WalltimeError, DFTConfig
from mlip_autopipec.physics.dft.recovery import RecoveryHandler


def test_recovery_scf_strategy():
    handler = RecoveryHandler()

    # Attempt 1: Reduce mixing beta
    params = {}
    new_params = handler.apply_fix(params, SCFError("Convergence failed"), attempt=1)
    assert new_params["mixing_beta"] == 0.3

    # Attempt 2: Further reduce mixing beta
    new_params = handler.apply_fix(params, SCFError("Convergence failed"), attempt=2)
    assert new_params["mixing_beta"] == 0.1

    # Attempt 3: Increase smearing (Methfessel-Paxton is default usually, try others)
    # Or just change mixing mode.
    # Let's assume strategy: 1->beta=0.3, 2->beta=0.1, 3->smearing='mv', degauss=0.02
    new_params = handler.apply_fix(params, SCFError("Convergence failed"), attempt=3)
    assert new_params["smearing"] == "mv"
    assert new_params["degauss"] == 0.02

def test_recovery_memory_strategy():
    handler = RecoveryHandler()

    # Memory error -> Reduce parallelization or change algo?
    # Usually we can't fix memory easily without changing resources (which is outside config).
    # But maybe we can reduce ecutwfc? (Dangerous for accuracy).
    # Or change diagonalization to 'cg' (conjugate gradient) which might use less memory than davidson?

    new_params = handler.apply_fix({}, MemoryError("OOM"), attempt=1)
    assert new_params.get("diagonalization") == "cg"

def test_recovery_unknown_error():
    handler = RecoveryHandler()

    # If generic error, maybe try robust settings?
    with pytest.raises(ValueError):
        # If we run out of strategies or don't know the error
        handler.apply_fix({}, Exception("Unknown"), attempt=1)

def test_max_retries():
    handler = RecoveryHandler()
    # Assuming we define max strategies for SCF
    # If attempt > max_strategies, raise?
    # QERunner handles max_retries loop.
    # But RecoveryHandler should return None or raise if no more fixes.

    # Let's say we have 3 fixes for SCF. Attempt 4 should fail.
    with pytest.raises(ValueError, match="No more recovery strategies"):
        handler.apply_fix({}, SCFError("Fail"), attempt=99)
