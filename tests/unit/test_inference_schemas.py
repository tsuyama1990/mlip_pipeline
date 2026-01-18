from pathlib import Path

import pytest
from pydantic import ValidationError

from mlip_autopipec.config.schemas.inference import InferenceConfig, InferenceResult


def test_inference_config_valid(tmp_path):
    potential_file = tmp_path / "model.yace"
    potential_file.touch()

    config = InferenceConfig(
        temperature=300.0,
        potential_path=potential_file
    )
    assert config.ensemble == "nvt"
    assert config.steps == 10000
    assert config.uq_threshold == 5.0

def test_inference_config_invalid_potential_path():
    with pytest.raises(ValidationError) as excinfo:
        InferenceConfig(
            temperature=300.0,
            potential_path="non_existent.yace"
        )
    assert "Path does not point to a file" in str(excinfo.value)

def test_inference_config_negative_temp(tmp_path):
    potential_file = tmp_path / "model.yace"
    potential_file.touch()
    with pytest.raises(ValidationError):
        InferenceConfig(
            temperature=-10.0,
            potential_path=potential_file
        )

def test_inference_result_valid():
    res = InferenceResult(
        succeeded=True,
        final_structure=Path("final.xyz"),
        uncertain_structures=[Path("dump_1.xyz")],
        max_gamma_observed=1.2
    )
    assert res.succeeded is True
    assert len(res.uncertain_structures) == 1
