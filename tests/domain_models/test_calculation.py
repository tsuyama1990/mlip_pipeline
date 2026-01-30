from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models.calculation import (
    DFTConfig,
    DFTResult,
    RecoveryConfig,
)


def test_dft_config_defaults():
    config = DFTConfig(
        command=["pw.x"],
        pseudopotentials={"Si": Path("Si.upf")},
        ecutwfc=30.0,
        kspacing=0.05,
    )
    assert config.timeout == 3600
    assert config.recovery.max_retries == 3
    assert "SCFError" in config.recovery.strategies


def test_dft_result_validation():
    # Valid result
    res = DFTResult(
        energy=-100.0,
        forces=np.zeros((2, 3)),
        stress=np.zeros((3, 3)),
    )
    assert res.converged

    # Invalid forces shape
    with pytest.raises(ValidationError):
        DFTResult(
            energy=-100.0,
            forces=np.zeros((2, 2)),  # Wrong shape
        )

    # Invalid stress shape
    with pytest.raises(ValidationError):
        DFTResult(
            energy=-100.0,
            forces=np.zeros((2, 3)),
            stress=np.zeros((4, 4)),  # Wrong shape
        )

    # Valid Voigt stress
    res_voigt = DFTResult(
        energy=-100.0,
        forces=np.zeros((2, 3)),
        stress=np.zeros((6,)),
    )
    assert res_voigt.stress.shape == (6,)


def test_recovery_config_strategies():
    config = RecoveryConfig()
    strategies = config.strategies
    assert isinstance(strategies, dict)
    assert "SCFError" in strategies
    assert isinstance(strategies["SCFError"], list)
    assert len(strategies["SCFError"]) > 0
