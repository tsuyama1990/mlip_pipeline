from pathlib import Path

import pytest
from pydantic import ValidationError

from mlip_autopipec.config.schemas.dft import DFTConfig
from mlip_autopipec.data_models.dft_models import DFTErrorType, DFTResult


def test_dft_config_valid():
    config = DFTConfig(
        command="mpirun -np 8 pw.x",
        pseudo_dir=Path("/path/to/pseudos"),
        timeout=7200,
        recoverable=False,
        max_retries=3,
    )
    assert config.command == "mpirun -np 8 pw.x"
    assert config.pseudo_dir == Path("/path/to/pseudos")
    assert config.timeout == 7200
    assert config.recoverable is False
    assert config.max_retries == 3


def test_dft_config_defaults():
    config = DFTConfig(pseudo_dir=Path("/tmp"))
    assert config.command == "mpirun -np 4 pw.x"
    assert config.timeout == 3600
    assert config.recoverable is True
    assert config.max_retries == 5


def test_dft_result_valid():
    result = DFTResult(
        uid="job-123",
        energy=-1234.5,
        forces=[[0.0, 0.0, 0.1]] * 10,
        stress=[[0.0] * 3] * 3,
        succeeded=True,
        wall_time=120.5,
        parameters={"ecutwfc": 60},
        final_mixing_beta=0.7,
    )
    assert result.uid == "job-123"
    assert result.energy == -1234.5
    assert len(result.forces) == 10
    assert result.succeeded is True


def test_dft_result_invalid_forces():
    with pytest.raises(ValidationError):
        DFTResult(
            uid="job-123",
            energy=-10.0,
            forces=[
                [0.0, 0.1]
            ],  # Invalid shape, expected Nx3 but this is 1x2 if interpreted as such? No, it's a list of list.
            # Wait, forces is List[List[float]]. Inner list should be checked?
            # The model doesn't enforce inner list length unless I add a validator.
            # SPEC says "Nx3 array".
            stress=[[0.0] * 3] * 3,
            succeeded=True,
            wall_time=10.0,
            parameters={},
            final_mixing_beta=0.7,
        )


def test_dft_error_type_enum():
    assert DFTErrorType.CONVERGENCE_FAIL == "CONVERGENCE_FAIL"
    assert DFTErrorType.NONE == "NONE"
