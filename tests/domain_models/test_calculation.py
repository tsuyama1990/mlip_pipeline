from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models.calculation import DFTConfig, DFTResult


def test_dft_config_valid():
    config = DFTConfig(
        command="pw.x",
        pseudopotentials={"Si": Path("Si.pbe.UPF")},
        ecutwfc=30.0,
        kspacing=0.04
    )
    assert config.command == "pw.x"
    assert config.kspacing == 0.04


def test_dft_result_valid():
    res = DFTResult(
        energy=-100.0,
        forces=np.zeros((2, 3)),
        stress=np.zeros((3, 3))
    )
    assert res.energy == -100.0


def test_dft_result_invalid_forces():
    with pytest.raises(ValidationError):
        DFTResult(
            energy=-100.0,
            forces=np.zeros((5,)), # Wrong shape
            stress=np.zeros((3, 3))
        )

def test_dft_result_invalid_stress():
    with pytest.raises(ValidationError):
        DFTResult(
            energy=-100.0,
            forces=np.zeros((2, 3)),
            stress=np.zeros((2, 2)) # Wrong shape
        )
