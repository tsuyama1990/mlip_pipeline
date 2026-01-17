import pytest
from pydantic import ValidationError

from mlip_autopipec.data_models.dft_models import DFTResult


def test_dft_result_validation():
    # Valid
    DFTResult(
        uid="id", energy=-1.0,
        forces=[[0.0, 0.0, 0.0]],
        stress=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        succeeded=True, wall_time=1.0, parameters={}, final_mixing_beta=0.7
    )

    # Invalid Forces Shape
    with pytest.raises(ValidationError) as excinfo:
        DFTResult(
            uid="id", energy=-1.0,
            forces=[[0.0, 0.0]], # 2D but inner wrong
            stress=[[0.0]*3]*3,
            succeeded=True, wall_time=1.0, parameters={}, final_mixing_beta=0.7
        )
    assert "Forces must have a shape of (N_atoms, 3)" in str(excinfo.value)

    # Invalid Stress Shape (Outer)
    with pytest.raises(ValidationError) as excinfo:
        DFTResult(
            uid="id", energy=-1.0,
            forces=[[0.0, 0.0, 0.0]],
            stress=[[0.0]*3]*2, # 2x3
            succeeded=True, wall_time=1.0, parameters={}, final_mixing_beta=0.7
        )
    assert "Stress tensor must be 3x3" in str(excinfo.value)

    # Invalid Stress Shape (Inner)
    with pytest.raises(ValidationError) as excinfo:
        DFTResult(
            uid="id", energy=-1.0,
            forces=[[0.0, 0.0, 0.0]],
            stress=[[0.0]*2]*3, # 3x2
            succeeded=True, wall_time=1.0, parameters={}, final_mixing_beta=0.7
        )
    assert "Stress tensor must be 3x3" in str(excinfo.value)
