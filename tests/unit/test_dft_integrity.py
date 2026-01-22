import pytest
from pydantic import ValidationError

from mlip_autopipec.data_models.dft_models import DFTResult


def test_dft_result_valid():
    res = DFTResult(
        uid="1", energy=-1.0, forces=[[0.0, 0.0, 0.0]], stress=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        succeeded=True, wall_time=1.0, parameters={}, final_mixing_beta=0.7
    )
    assert res.succeeded

def test_dft_result_invalid_forces_shape():
    # 2 atoms, but one has 2 components
    with pytest.raises(ValidationError, match="Forces must have a shape"):
        DFTResult(
            uid="1", energy=-1.0, forces=[[0.0, 0.0, 0.0], [0.0, 0.0]],
            stress=[[0.0, 0.0, 0.0]]*3, succeeded=True, wall_time=1.0, parameters={}, final_mixing_beta=0.7
        )

def test_dft_result_invalid_stress_shape():
    # 2x3 tensor instead of 3x3
    with pytest.raises(ValidationError, match="Stress tensor must be 3x3"):
        DFTResult(
            uid="1", energy=-1.0, forces=[[0.0, 0.0, 0.0]],
            stress=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            succeeded=True, wall_time=1.0, parameters={}, final_mixing_beta=0.7
        )

    # 3 rows but one row has 2 cols
    with pytest.raises(ValidationError, match="Stress tensor must be 3x3"):
        DFTResult(
            uid="1", energy=-1.0, forces=[[0.0, 0.0, 0.0]],
            stress=[[0.0, 0.0, 0.0], [0.0, 0.0], [0.0, 0.0, 0.0]],
            succeeded=True, wall_time=1.0, parameters={}, final_mixing_beta=0.7
        )
